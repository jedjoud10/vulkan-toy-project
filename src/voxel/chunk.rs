use std::collections::VecDeque;
use fixedbitset::FixedBitSet;
use crate::voxel::{sparse::ChunkLevelAccelerationStructureNode, util::offset_to_index};

pub const CHUNK_SIZE: usize = 64;
pub const CHUNK_VOLUME: usize = 64*64*64;

pub enum ChunkData {
    Full,
    Empty,
    Partial(FixedBitSet)
}


// invariant: ChunkData MUST be in correct state
// it cannot be in the "partial" state if it contains a fully cleared or fully set bitset
pub struct Chunk {
    pub position: vek::Vec3<u32>,
    pub voxel_data: ChunkData,
    pub sparse_representation: Vec<ChunkLevelAccelerationStructureNode>,
    pub bounds: vek::Aabb<u32>,
}

impl Chunk {
    pub fn new(position: vek::Vec3<u32>, data: FixedBitSet) -> Self {
        let voxel_data = if data.is_full() {
            ChunkData::Full
        } else if data.is_clear() {
            ChunkData::Empty
        } else {
            ChunkData::Partial(data)
        };

        Self {
            bounds: vek::Aabb::default(),
            position,
            voxel_data,
            sparse_representation: Vec::new(),
        }
    }

    /*
    pub fn set(&mut self, position: vek::Vec3<usize>, voxel: bool) { 
        assert!(position.cmpge(&vek::Vec3::<usize>::zero()).reduce_and());
        assert!(position.cmplt(&vek::Vec3::<usize>::broadcast(CHUNK_SIZE)).reduce_and());
        let index = offset_to_index(position, CHUNK_SIZE);
        
        match (&mut self.voxel_data, voxel) {
            // do nothing
            (ChunkData::Empty, false) | (ChunkData::Full, true) => {},
            
            // create bitset from FULL and remove bit
            (ChunkData::Full, false) => {
                let mut bitset = FixedBitSet::with_capacity_and_blocks(CHUNK_VOLUME, std::iter::repeat(usize::MAX));
                bitset.set(index, false);
            },
            
            // create bitset from EMPTY and remove bit
            (ChunkData::Empty, true) => {
                // starts empty, with all bits set to false
                let mut bitset = FixedBitSet::with_capacity(CHUNK_VOLUME);
                bitset.set(index, false);
            }

            // set normally, but might need to change type of voxel data if all bits where set to zero / one 
            (ChunkData::Partial(fixed_bit_set), set) => {
                fixed_bit_set.set(index, set);

                if fixed_bit_set.is_clear() {
                    self.voxel_data = ChunkData::Empty;
                } else if fixed_bit_set.is_full() {
                    self.voxel_data = ChunkData::Full
                }
            },
        }
    }

    pub fn get(&self, i: usize) -> bool {
        match &self.voxel_data {
            ChunkData::Full => true,
            ChunkData::Empty => false,
            ChunkData::Partial(fixed_bit_set) => fixed_bit_set[i],
        }
    }
    */

    pub fn is_full(&self) -> bool {
        matches!(self.voxel_data, ChunkData::Full)
    }

    pub fn is_empty(&self) -> bool {
        matches!(self.voxel_data, ChunkData::Empty)
    }

    pub fn rebuild(&mut self) {
        (self.sparse_representation, self.bounds) = chunk_to_sparse(&self.voxel_data, self.position);
    }
}


fn chunk_to_sparse(data: &ChunkData, chunk_position: vek::Vec3<u32>) -> (Vec<ChunkLevelAccelerationStructureNode>, vek::Aabb<u32>) {
    let full_world_space_bounds = vek::Aabb::<u32> {
        min: chunk_position * CHUNK_SIZE as u32,
        max: chunk_position * CHUNK_SIZE as u32 + CHUNK_SIZE as u32,
    };

    let data = match data {
        ChunkData::Full => return (vec![ChunkLevelAccelerationStructureNode { bounds: full_world_space_bounds, children: None, full: true }], full_world_space_bounds),
        ChunkData::Empty => return (vec![ChunkLevelAccelerationStructureNode { bounds: vek::Aabb::default(), children: None, full: false }], vek::Aabb::default()),
        ChunkData::Partial(data) => data 
    };
    
    const CHUNK_64_HEIGHT_4_TREE: u32 = 4;

    // bottom up approach: do multiple passes, starting from the bottom
    // this will generate the "mips" of the chunk
    // this can be parallelized
    // this can be optimized further if using morton encoding, because then, groups of 4x4x4 nodes are a contiguous slice of 64 bits. We can use batch operations to speed that up instead of keeping the inner most 3 loops 
    let mut any_mips = [const { FixedBitSet::new() }; CHUNK_64_HEIGHT_4_TREE as usize];
    let mut all_mips = [const { FixedBitSet::new() }; CHUNK_64_HEIGHT_4_TREE as usize];
    let mut all_bounds = [const { Vec::<vek::Aabb<u32>>::new() }; CHUNK_64_HEIGHT_4_TREE as usize];
    
    // of course we must store the initial mip lol
    any_mips[0] = data.clone();
    all_mips[0] = data.clone();
    
    for pass in 1..4usize {
        let mip_size = 64usize / (1 << ((pass)*2)); // 16, 4, 1
        let voxel_size = 64usize / mip_size; // 4, 16, 64

        log::debug!("pass {pass}, mip size: {mip_size}");

        let previous_mip_size = mip_size * 4;
        let previous_voxel_size = voxel_size / 4;

        let previous_any_mip = &any_mips[pass-1];
        let previous_all_mip = &all_mips[pass-1];
        let previous_all_bounds = &all_bounds[pass-1];

        log::debug!("pass: {pass}, num PREVIOUS any mip bits set: {}", previous_any_mip.count_ones(..));
        log::debug!("pass: {pass}, num PREVIOUS all mip bits set: {}", previous_all_mip.count_ones(..));

        let mip_volume = (mip_size as usize).pow(3);
        let mut next_any_mip = FixedBitSet::with_capacity(mip_volume);
        let mut next_all_mip = FixedBitSet::with_capacity(mip_volume);
        let mut next_all_bounds = vec![vek::Aabb::<u32>::default(); mip_volume];

        for x in 0..mip_size {
            for y in 0..mip_size {
                for z in 0..mip_size {
                    let mut any = false;
                    let mut all = true;
                    let offset = vek::Vec3::new(x,y,z);

                    let mut local_bound = vek::Aabb::<u32> {
                        min: vek::Vec3::broadcast(u32::MAX),
                        max: vek::Vec3::broadcast(0)
                    };
                    for local_x in 0..4 {
                        for local_y in 0..4 {
                            for local_z in 0..4 {
                                let local_offset = vek::Vec3::new(local_x, local_y, local_z);
                                let position: vek::Vec3<usize> = local_offset + offset * 4;
                                let i = offset_to_index(position, previous_mip_size); 
                                any |= previous_any_mip[i];
                                all &= previous_all_mip[i];

                                if previous_any_mip[i] {
                                    if pass == 1 {
                                        // easy case for first pass
                                        let chunk_space_position = position * previous_voxel_size;
                                        local_bound.expand_to_contain_point(chunk_space_position.as_::<u32>());
                                        local_bound.expand_to_contain_point(chunk_space_position.as_::<u32>()+1);
                                    } else {
                                        // use previous pass bounds...
                                        local_bound.expand_to_contain(previous_all_bounds[i]);
                                    }
                                }
                            }
                        }
                    }

                    //let i = (x + y * 4 + z * 4 * 4) as usize; 
                    let i = offset_to_index(offset, mip_size); 
                    next_any_mip.set(i, any);
                    next_all_mip.set(i, all);
                    next_all_bounds[i] = local_bound;
                }
            }
        }

        log::debug!("pass: {pass}, num next any mip bits set: {}", next_any_mip.count_ones(..));
        log::debug!("pass: {pass}, num next all mip bits set: {}", next_all_mip.count_ones(..));
        

        any_mips[pass] = next_any_mip;
        all_mips[pass] = next_all_mip;
        all_bounds[pass] = next_all_bounds;
    }

    let chunk_local_space_bound = (&all_bounds[3])[0];
    log::debug!("chunk local space bound: min:{}, max:{}", chunk_local_space_bound.min, chunk_local_space_bound.max);
    let chunk_world_space_bound = vek::Aabb::<u32> {
        min: chunk_local_space_bound.min + chunk_position * CHUNK_SIZE as u32,
        max: chunk_local_space_bound.max + chunk_position * CHUNK_SIZE as u32,
    };

    // start top down and create some nodes
    // we can inline the nodes in any fashion we want in the array, as long as their indices match up
    // we can write them in BFS or DFS order. does not matter
    (convert_mips_to_nodes(chunk_position * CHUNK_SIZE as u32, &all_mips, &any_mips, &all_bounds), chunk_world_space_bound)
}


struct NotSoSimpleTraversalNode {
    mip_index: usize,
    index_within_mip: usize,
    height: u32,
    origin: vek::Vec3<u32>,
    local_chunk_bounds: vek::Aabb<u32>,
}

// mip 0 is bottom most mip
// mip N-1 is one node (top mip)
pub fn convert_mips_to_nodes<const MIP_COUNT: usize>(chunk_world_space_origin: vek::Vec3<u32>, all_mips: &[FixedBitSet; MIP_COUNT], any_mips: &[FixedBitSet; MIP_COUNT], all_bounds: &[Vec<vek::Aabb<u32>>; MIP_COUNT]) -> Vec<ChunkLevelAccelerationStructureNode> {
    let mut queue = VecDeque::<NotSoSimpleTraversalNode>::new();
    queue.push_back(NotSoSimpleTraversalNode { mip_index: MIP_COUNT-1, index_within_mip: 0, height: MIP_COUNT as u32-1, origin: vek::Vec3::zero(), local_chunk_bounds: all_bounds[MIP_COUNT - 1][0] });

    let mut nodes = Vec::<ChunkLevelAccelerationStructureNode>::new();

    let mut estimated_next_index = 0usize;

    while let Some(NotSoSimpleTraversalNode { mip_index, index_within_mip, height, origin, local_chunk_bounds }) = queue.pop_front() {
        let voxel_size: u32 = 4u32.pow(height);
        let mip_size: usize = CHUNK_SIZE / voxel_size as usize;


        let world_space_bounds = vek::Aabb::<u32> {
            min: local_chunk_bounds.min + chunk_world_space_origin,
            max: local_chunk_bounds.max + chunk_world_space_origin,
        };

        let is_node_any = (any_mips[mip_index])[index_within_mip];
        let is_node_all = (all_mips[mip_index])[index_within_mip];

        // testing purposes
        if mip_index == 0 {
            nodes.push(ChunkLevelAccelerationStructureNode {
                bounds: world_space_bounds,
                children: None,
                full: is_node_all,
            });
            continue;
        }


        //log::debug!("mip index: {mip_index}, mip size: {mip_size}, voxel size: {voxel_size}, index within mip: {index_within_mip}, height: {height}, origin: {origin}, is node any: {is_node_any}, is node all: {is_node_all}");

        let children = if height == 0 {
            None
        } else {
            if is_node_all {
                None
            } else {
                if is_node_any {
                    //log::debug!("node any");
                    let mut flat_node_children: Box<[Option<usize>; 64]> = Box::new([Option::<usize>::None; 64]);

                    let next_mip_size = mip_size * 4;

                    // node has children, add them to queue
                    for child_index in 0..(4*4*4) {
                        // if the child is present, then it must also have a node
                        //log ::debug!("checking child...");
                        let child_offset = super::util::child_index_to_child_offset(child_index);

                        // calculate child origin in CHUNK SPACE
                        let child_origin_chunk_space = origin + child_offset.as_::<u32>() * (voxel_size / 4);
                        
                        // child index in next mip is in NEXT MIP SPACE
                        let child_origin_next_mip_space = ((origin.as_::<usize>() / (voxel_size as usize / 4)) + child_offset).as_::<usize>();
                        //log::debug!("child origin chunk space: {child_origin_chunk_space}, child offset: {child_offset}, child position in next mip: {}", child_origin_next_mip_space);
                        let child_index_in_next_mip = offset_to_index(child_origin_next_mip_space, next_mip_size);
                        //log::debug!("child index in next mip: {child_index_in_next_mip}");

                        assert!(child_index_in_next_mip < (next_mip_size * next_mip_size * next_mip_size));
                        assert!((next_mip_size * next_mip_size * next_mip_size) == any_mips[mip_index-1].len());

                        if (any_mips[mip_index-1])[child_index_in_next_mip] {
                            if mip_index > 1 {
                                //log::debug!("add child push back!!");
                                queue.push_back(NotSoSimpleTraversalNode {
                                    mip_index: mip_index-1,
                                    index_within_mip: child_index_in_next_mip,
                                    height: height-1,
                                    origin: child_origin_chunk_space,
                                    local_chunk_bounds: all_bounds[mip_index-1][child_index_in_next_mip]
                                });
                                estimated_next_index += 1;
                                flat_node_children[child_index] = Some(estimated_next_index);
                            } else {
                                flat_node_children[child_index] = Some(usize::MAX);
                            }
                        }
                    }

                    Some(flat_node_children)
                } else {
                    None
                }
            }
        };

        // add node to flat node list
        nodes.push(ChunkLevelAccelerationStructureNode {
            bounds: world_space_bounds,
            children,
            full: is_node_all,
        });
    }

    log::debug!("num generated chunk level nodes {}", nodes.len());
    
    nodes
}


/*
KEEPING THIS HERE IF WE WANT TO OPTIMIZE CHUNK ACCEL STRUCTURE REBUILDING SPEED (INCREMENTAL)



    fn traverse<A: FnMut(&mut Vec<FlatNode>, &TopDownTraversalNode, u32) -> ControlFlow<(), ()>, B: FnMut(&mut Vec<FlatNode>, &TopDownTraversalNode, usize, u32) -> ControlFlow<(), ()>>(&mut self, pos: vek::Vec3<u32>, target_height: u32, mut prefire_callback: A, mut postfire_callback: B) {
        assert!(pos.cmpge(&vek::Vec3::broadcast(0u32)).reduce_and());
        assert!(pos.cmplt(&vek::Vec3::broadcast(TOTAL_SIZE)).reduce_and());
        
        let mut current = Some(TopDownTraversalNode { index: 0, height: SVO_DEPTH-1, origin: vek::Vec3::zero() });

        while let Some(node) = current.take() {
            let size = 1 << (node.height*2);
            
            log::trace!("depth: {}, origin: {}, size: {}", node.height, node.origin, size);
            let child_offset = (pos - node.origin) / vek::Vec3::broadcast(size);
            log::trace!("child offset: {}", child_offset);

            assert!(child_offset.cmpge(&vek::Vec3::broadcast(0u32)).reduce_and());
            assert!(child_offset.cmplt(&vek::Vec3::broadcast(4u32)).reduce_and());
            let (x, y, z) = child_offset.into_tuple();

            let child_index_relative = x + y * 4 + z * 4 * 4;

            self.nodes[node.index].bounds.expand_to_contain(vek::Aabb {
                min: pos,
                max: pos+1,
            });


            let r = prefire_callback(&mut self.nodes, &node, child_index_relative);
            if r.is_break() {
                break;
            }

            let child_index_absolute = if let Some(idx) = self.nodes[node.index].children.as_mut().unwrap()[child_index_relative as usize] {
                self.nodes[idx].bounds.expand_to_contain(vek::Aabb {
                    min: pos,
                    max: pos+1,
                });
                idx
            } else {
                self.nodes.push(FlatNode {
                    children: None,
                    full: false,
                    bounds: vek::Aabb {
                        min: pos,
                        max: pos+1,
                    },
                });
                let idx = self.nodes.len()-1;
                self.nodes[node.index].children.as_mut().unwrap()[child_index_relative as usize] = Some(idx);
                idx
            };
            
            let r = postfire_callback(&mut self.nodes, &node, child_index_absolute, child_index_relative);
            if r.is_break() {
                break;
            }

            if node.height > target_height {
                current = Some(TopDownTraversalNode {
                    index: child_index_absolute,
                    height: node.height - 1,
                    origin: child_offset * size + node.origin,
                });
            }
        }
    } 
    
    pub fn set(&mut self, pos: vek::Vec3<u32>, voxel: bool) {
        if pos.cmplt(&vek::Vec3::broadcast(0u32)).reduce_or() || pos.cmpge(&vek::Vec3::broadcast(TOTAL_SIZE)).reduce_or() {
            log::trace!("voxel at {pos} is OOB, ignoring");
            return;
        }

        log::trace!("setting voxel at {pos} to {voxel}");

        let mut path = Vec::<BottomUpPath>::new();

        let prefire_callback = |nodes: &mut Vec<FlatNode>, node: &TopDownTraversalNode, child_index_relative: u32| -> ControlFlow<(), ()> {
            // no need to do anything if we are trying to add a voxel to an already full sub-tree
            if voxel && nodes[node.index].full {
                return ControlFlow::Break(());
            }

            nodes[node.index].children.get_or_insert_with(|| Box::new([const { None }; 64]));

            // if we are removing a voxel from a full node, we must add all of its OTHER children that must be full, except the one we are modifying
            if !voxel && nodes[node.index].full {
                for i in 0..64 {
                    if i != child_index_relative as usize /* && self.nodes[node.index].children.as_ref().unwrap()[i].is_none() */ {
                        nodes.push(FlatNode {
                            children: None,
                            full: true,
                            bounds: vek::Aabb::new_empty(vek::Vec3::zero())
                        });
                        let idx = nodes.len()-1;
                        nodes[node.index].children.as_mut().unwrap()[i] = Some(idx);
                    }
                }
            }

            ControlFlow::Continue(())
        };


        let postfire_callback = |nodes: &mut Vec<FlatNode>, node: &TopDownTraversalNode, child_index_absolute: usize, child_index_relative: u32| -> ControlFlow<(), ()> {
            path.push(BottomUpPath { parent_index: node.index, child_index_relative, child_index_absolute });

            // set bottom most child as FULL
            if node.height == 0 && voxel {
                nodes[child_index_absolute].full = true;
            }
            
            // remove bottom most child
            if node.height == 0 && !voxel {            
                // TODO: also remove the node in the array, but that requires shifting all indices and recalculating them... :(
                // need to use a slotmap....
                nodes[node.index].children.as_mut().unwrap()[child_index_relative as usize].take().unwrap();
            }

            // if the parent (node.index) was full, then the just added node must ALSO be full so that we can recursively set its children as full until we reach the bottom
            if !voxel && nodes[node.index].full {
                nodes[node.index].full = false;
                nodes[child_index_absolute].full = true;
            }

            ControlFlow::Continue(())
        };

        self.traverse(pos, 0, prefire_callback, postfire_callback);

        // BOTTOM UP
        for (_, node) in path.iter().enumerate().rev() {
            if voxel {
                // recalculate node fullness based on children fullness
                if let Some(children) = self.nodes[node.child_index_absolute].children.as_ref() {
                    let full = children.iter().all(|child| child.as_ref().map(|c| self.nodes[*c ].full).unwrap_or_default());
                    self.nodes[node.child_index_absolute].full = full;
                }
            } else {
                // get rid of node if all children are missing
                let borrowed = &self.nodes[node.child_index_absolute];
                if (borrowed.children.is_none() && !borrowed.full) || borrowed.children.as_ref().map(|children| children.iter().all(|opt_child| opt_child.is_none())).unwrap_or_default() {
                    self.nodes[node.parent_index].children.as_mut().unwrap()[node.child_index_relative as usize] = None;
                }
            }
        }
    }


*/