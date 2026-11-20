

struct Page {
    in_use: bool,
}

// contains a contiguous list of pages
// when we do an allocation, we try to fit it into some contiguous "free" pages
// this suffers from both internal and external fragmentation
// always allocates in entire pages
pub struct SubBufferAllocator {
    // TODO (OPTIMIZATION): use a bitmask to speed up page retrieval
    // TODO (OPTIMIZATION): use multiple layers / page sizes to reduce fragmentation
    pages: Vec<Page>,
    page_size: usize,
    currently_allocated_bytes: usize,
    total_buffer_size: usize,
}

pub struct SubBufferAllocatorStats {
    pub total_buffer_size: usize,
    pub total_allocated_bytes: usize,
    pub internal_frag_bytes: usize,
    pub pages_allocated: usize,
    pub page_size: usize,
}

impl std::fmt::Display for SubBufferAllocatorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total = bytesize::ByteSize::b(self.total_allocated_bytes as u64);
        let wasted = bytesize::ByteSize::b(self.internal_frag_bytes as u64);
        let pages = self.pages_allocated;
        let unused = bytesize::ByteSize::b(self.total_buffer_size as u64 - self.total_allocated_bytes as u64);
        let page_size = bytesize::ByteSize::b(self.page_size as u64);
        
        write!(f, "total: {total}, wasted (internal frag): {wasted}, unused: {unused}, pages: {pages}, page size: {page_size}")
    }
}

pub struct SubBufferAllocation {
    pub start_byte_offset: usize,
    pub start_page: usize,
    pub page_count: usize,
    pub byte_size: usize
}


impl SubBufferAllocator {
    pub fn new(total_buffer_size: usize, page_size: usize) -> Self {
        log::debug!("creating sub buffer allocator for {} and page size {}", bytesize::ByteSize::b(total_buffer_size as u64), bytesize::ByteSize::b(page_size as u64));
        let num_pages = total_buffer_size.div_ceil(page_size);
        log::debug!("number of pages: {}", num_pages);
        
        
        Self {
            pages: (0..num_pages).into_iter().map(|_| Page { in_use: false }).collect(),
            page_size,
            total_buffer_size,
            currently_allocated_bytes: 0,
        }
    }

    pub fn allocate(&mut self, byte_size: usize) -> Option<SubBufferAllocation> {
        let num_contiguous_pages_required = byte_size.div_ceil(self.page_size);
        let num_pages = self.pages.len();

        // TODO (OPTIMIZATION): literally do anything BUT this. lmao
        for start_page in 0..num_pages {
            let end = (start_page+num_contiguous_pages_required);

            if end > num_pages {
                break;
            }

            let slice = &mut self.pages[start_page..end]; 
            let all_free = slice.iter().all(|x| !x.in_use);
            let start_byte_offset = start_page * self.page_size; 

            if all_free {
                // set as used
                slice.iter_mut().for_each(|x| x.in_use = true);

                self.currently_allocated_bytes += byte_size; 

                return Some(SubBufferAllocation {
                    start_byte_offset,
                    start_page,
                    byte_size,
                    page_count: num_contiguous_pages_required,
                });
            }
        }
    
        None
    }

    pub fn total_internal_fragmentation(&self) -> usize {
        let num_pages_in_use = self.pages.iter().filter(|x| x.in_use).count();
        let total_page_bytes = num_pages_in_use * self.page_size;
        return total_page_bytes - self.currently_allocated_bytes;
    }

    pub fn stats(&self) -> SubBufferAllocatorStats {
        SubBufferAllocatorStats {
            total_buffer_size: self.total_buffer_size,
            total_allocated_bytes: self.currently_allocated_bytes,
            internal_frag_bytes: self.total_internal_fragmentation(),
            pages_allocated: self.pages.iter().filter(|x| x.in_use).count(),
            page_size: self.page_size,
        }
    }

    pub fn free(&mut self, alloc: SubBufferAllocation) {        
        let end = (alloc.start_page+alloc.page_count).min(self.pages.len());
        let slice = &mut self.pages[(alloc.start_page)..end]; 
        slice.iter_mut().for_each(|x| x.in_use = false);
        self.currently_allocated_bytes -= alloc.byte_size;
    }
}

#[cfg(test)]
mod tests {
    pub use super::*;

    #[test]
    fn test_allocator_internal_fragmentation() {
        let mut allocator = SubBufferAllocator::new(1024, 32);
        let a1 = allocator.allocate(4).unwrap();
        assert_eq!(a1.page_count, 1);
        assert_eq!(a1.start_page, 0);

        let a2 = allocator.allocate(4).unwrap();
        assert_eq!(a2.page_count, 1);
        assert_eq!(a2.start_page, 1);

        let a3 = allocator.allocate(4).unwrap();
        assert_eq!(a3.page_count, 1);
        assert_eq!(a3.start_page, 2);

        assert!(a3.start_byte_offset != a2.start_byte_offset);
        assert!(a3.start_byte_offset != a1.start_byte_offset);
        assert!(a1.start_byte_offset != a3.start_byte_offset);
    }

    #[test]
    fn test_allocator_one_page() {
        let mut allocator = SubBufferAllocator::new(1024, 32);
        let a1 = allocator.allocate(32).unwrap();
        assert_eq!(a1.page_count, 1);

        let a2 = allocator.allocate(32).unwrap();
        assert_eq!(a2.page_count, 1);

        let a3 = allocator.allocate(32).unwrap();
        assert_eq!(a3.page_count, 1);

        assert!(a3.start_byte_offset != a2.start_byte_offset);
        assert!(a3.start_byte_offset != a1.start_byte_offset);
        assert!(a1.start_byte_offset != a3.start_byte_offset);
    }

    #[test]
    fn test_allocator_multiple_pages() {
        let mut allocator = SubBufferAllocator::new(1024, 32);
        let a1 = allocator.allocate(34).unwrap();
        assert_eq!(a1.page_count, 2);

        let a2 = allocator.allocate(60).unwrap();
        assert_eq!(a2.page_count, 2);

        let a3 = allocator.allocate(65).unwrap();
        assert_eq!(a3.page_count, 3);

        assert!(a3.start_byte_offset != a2.start_byte_offset);
        assert!(a3.start_byte_offset != a1.start_byte_offset);
        assert!(a1.start_byte_offset != a3.start_byte_offset);
    }

    
    #[test]
    fn test_allocator_entire_allocator() {
        let mut allocator = SubBufferAllocator::new(1024, 32);
        let a1 = allocator.allocate(1024).unwrap();
        assert_eq!(a1.page_count, allocator.pages.len());
        assert_eq!(a1.byte_size, 1024);
        
        let should_fail = allocator.allocate(60);
        assert!(should_fail.is_none());
    }
}
