use std::{
    borrow::Cow, collections::HashMap, env, fs::{self, DirEntry, File}, io::Write, path::{Path, PathBuf}
};

use shader_slang::*;

fn load_module(session: &Session, file_name: &str, compiled: &mut HashMap<String, Cow<[u8]>>) {
    let file_name_with_extension = &format!("{file_name}.slang");

    // ok this caching idea failed pretty quickly because if we want to get the dependencies of each file we have to load the module already, which is the expensive part
    // we might need to implement our own dependency parser so that we can avoid feeding it to sland immediately.
    log::debug!("loading module for file '{}'", file_name_with_extension);
    let module: Module = session.load_module(file_name_with_extension).unwrap();
    
    if module.entry_point_count() == 0 {
        return;
    }

    let mut component_types = Vec::<ComponentType>::new();
    component_types.push(ComponentType::from(module.clone()));
    for entry_point in module.entry_points() {
        component_types.push(ComponentType::from(entry_point));
    }

    log::debug!("linking modules for file '{}'", file_name_with_extension);
    let program = session.create_composite_component_type(&component_types).unwrap();
    let linked_program = program.link().unwrap();

    // ERROR: for some reason `target_code` does not return Err, even when there is no valid target code
    // it just gives an undefined blob. program crashes when you try to `as_slice` it.
    // TODO: report as issue?
    let shader_bytecode = linked_program.target_code(0).unwrap();
    let raw = shader_bytecode.as_slice();

    compiled.insert(file_name.to_string(), Cow::from(raw.to_vec()));

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

pub fn compile_all_shaders(compiled: &mut HashMap<String, Cow<[u8]>>) {
    log::info!("compiling slang shaders");
    let global_session = GlobalSession::new().unwrap();

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

    // visit all the files inside the shaders/other folders
    let manifest_dir_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut shaders_path = manifest_dir_path.clone();
    shaders_path.push("shaders");

    log::info!("visiting shader directories");
    let mut entries = Vec::<DirEntry>::new();
    visit_dirs(&shaders_path, &mut entries);

    for entry in entries {
        let file_name = entry.file_name().into_string().unwrap();
        let file_name = file_name.split(".").next().unwrap();
        load_module(&session, file_name, compiled);
    }
}
