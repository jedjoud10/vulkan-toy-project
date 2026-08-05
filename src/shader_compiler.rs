use std::{
    borrow::Cow, cell::{Cell, LazyCell, OnceCell, RefCell}, collections::HashMap, env, fs::{self, DirEntry, File}, io::Write, path::{Path, PathBuf}
};

use rayon::iter::{IntoParallelIterator, Once, ParallelIterator};
use shader_slang::{CompileTarget, CompilerOptions, ComponentType, DebugInfoLevel, GlobalSession, Module, OptimizationLevel, Session, SessionDesc, TargetDesc};

type SlangResult<T> = shader_slang::Result<T>;
type CompiledEntry = SlangResult<Option<(String, Vec<u8>)>>;

thread_local! {
    static THREAD_LOCAL_GLOBAL_SESSION: OnceCell<GlobalSession> = OnceCell::new();
}

fn load_module(session: &Session, file_name: &str) -> CompiledEntry {
    let file_name_with_extension = &format!("{file_name}.slang");

    // ok this caching idea failed pretty quickly because if we want to get the dependencies of each file we have to load the module already, which is the expensive part
    // we might need to implement our own dependency parser so that we can avoid feeding it to sland immediately.
    log::debug!("loading module for file '{}'", file_name_with_extension);
    let module: Module = session.load_module(file_name_with_extension)?;
    
    // ERROR: for some reason `target_code` does not return Err, even when there is no valid target code
    // it just gives an undefined blob. program crashes when you try to `as_slice` it.
    // this is why we have this earlier check
    // TODO: report as issue?
    if module.entry_point_count() == 0 {
        return Ok(None);
    }


    let mut component_types = Vec::<ComponentType>::new();
    component_types.push(ComponentType::from(module.clone()));
    for entry_point in module.entry_points() {
        component_types.push(ComponentType::from(entry_point));
    }

    log::debug!("linking modules for file '{}'", file_name_with_extension);
    let program = session.create_composite_component_type(&component_types)?;
    let linked_program = program.link()?;
    let shader_bytecode = linked_program.target_code(0)?;
    let raw = shader_bytecode.as_slice();

    return Ok(Some((file_name.to_string(), raw.to_vec())));
    /*
    // write the file to the compiled shaders folder
    let mut path = PathBuf::from(compiled_shader_folder_path);
    path.push(format!("{file_name}.spv"));
    let mut file = File::create(&path).unwrap();
    file.write_all(shader_bytecode.as_slice()).unwrap();
    */
}

// https://doc.rust-lang.org/nightly/std/fs/fn.read_dir.html#examples
fn visit_dirs(dir: &Path, list: &mut Vec<DirEntry>) {
    if dir.is_dir() {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            if entry.path().is_dir() {
                visit_dirs(&entry.path(), list);
            } else {
                list.push(entry);
            }
        }
    }
}

// TODO: implement some sort of caching system that avoid recompiling shaders that have already been compiled
pub fn compile_all_shaders() -> Result<HashMap<String, Vec<u8>>, ()> {
    log::info!("compiling slang shaders");
    let start = std::time::Instant::now();

    // visit all the files inside the shaders/other folders
    let manifest_dir_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut shaders_path = manifest_dir_path.clone();
    shaders_path.push("shaders");

    log::info!("visiting shader directories");
    let mut entries = Vec::<DirEntry>::new();
    
    // HACK: no need to compile modules for utility files
    //visit_dirs(&shaders_path, &mut entries);
    for entry in fs::read_dir(shaders_path).unwrap() {
        let entry = entry.unwrap();
        if entry.path().is_file() {
            entries.push(entry);
        }
    }

    // running this on multiple threads does help, but we incur the cost of creating a GlobalSession for EACH thread
    // plus, shared files that are imported will be re-compiled. uggggghhh this sucks
    let compiled_entries = entries.into_par_iter().map(|entry| {
        THREAD_LOCAL_GLOBAL_SESSION.with(|value| {
            let global_session = value.get_or_init(|| GlobalSession::new().unwrap());

            // TODO: revert optimization level when vulkan-sdk ships with latest slang compiler bugfix for NonUniform indexing
            // nvm I still get the HWRT blocky group artifacts on my hw. AMD driver issue perhaps? could also be a code issue. I might be missing a NonUniform somewhere
            let session_options = CompilerOptions::default()
                .optimization(OptimizationLevel::None)
                .debug_information(DebugInfoLevel::Minimal)
                .obfuscate(false)
                .no_mangle(true)
                .disable_specialization(false)
                .vulkan_use_entry_point_name(true)
                .matrix_layout_row(true);

            
            let target_desc = TargetDesc::default().format(CompileTarget::Spirv);
            let targets = [target_desc];

            // TODO: replace with non-hard-coded version
            let search_paths = [c"shaders".as_ptr(), c"shaders/utils".as_ptr(), c"shaders/utils/noises".as_ptr()];

            let session_desc = SessionDesc::default()
                .targets(&targets)
                .search_paths(&search_paths)
                .options(&session_options);

            let session = global_session.create_session(&session_desc).unwrap();
            let file_name = entry.file_name().into_string().unwrap();
            let file_name = file_name.split(".").next().unwrap();

            load_module(&session, file_name)
        })
    }).collect::<Vec<CompiledEntry>>();

    let mut compiled = HashMap::<String, Vec<u8>>::new();

    for result in compiled_entries {

        match result {
            Ok(Some((file_name, bytecode))) => {
                compiled.insert(file_name, bytecode);
            },
            Err(err) => {
                log::error!("shader compilation error");
                log::error!("{}", err);
                return Err(());
            },
            _ => {}
        }
        
    }


    let end = std::time::Instant::now();

    log::info!("compiled {} shaders in {:.2}s", compiled.len(), (end-start).as_secs_f32());
    Ok(compiled)
}
