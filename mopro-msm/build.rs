use std::{env, path::Path, process::Command};
use walkdir::WalkDir;

const METAL_SHADER_DIR: &str = "src/msm/metal/shader/";

fn main() {
    compile_shaders();
    setup_rebuild();
}

fn compile_shaders() {
    let out_dir = env::var("OUT_DIR").unwrap();

    // List your Metal shaders here.
    let shaders = vec!["all.metal"];

    let mut air_files = vec![];

    // Step 1: Compile every shader to AIR format
    for shader in &shaders {
        let shader_path = Path::new(METAL_SHADER_DIR).join(shader);
        let air_output = Path::new(&out_dir).join(format!("{}.air", shader));

        let mut args = vec![
            "-sdk",
            get_sdk(),
            "metal",
            "-c",
            shader_path.to_str().unwrap(),
            "-o",
            air_output.to_str().unwrap(),
        ];

        if cfg!(feature = "profiling-release") {
            args.push("-frecord-sources");
        }

        // Compile shader into .air files
        let status = Command::new("xcrun")
            .args(&args)
            .status()
            .expect("Shader compilation failed");

        if !status.success() {
            panic!("Shader compilation failed for {}", shader);
        }

        air_files.push(air_output);
    }

    // Step 2: Link all the .air files into a Metallib archive
    let metallib_output = Path::new(&out_dir).join("msm.metallib");

    let mut metallib_args = vec![
        "-sdk",
        get_sdk(),
        "metal",
        "-o",
        metallib_output.to_str().unwrap(),
    ];

    if cfg!(feature = "profiling-release") {
        metallib_args.push("-frecord-sources");
    }

    for air_file in &air_files {
        metallib_args.push(air_file.to_str().unwrap());
    }

    let status = Command::new("xcrun")
        .args(&metallib_args)
        .status()
        .expect("Failed to link shaders into metallib");

    if !status.success() {
        panic!("Failed to link shaders into metallib");
    }

    let symbols_args = vec![
        "metal-dsymutil",
        "-flat",
        "-remove-source",
        metallib_output.to_str().unwrap(),
    ];

    let status = Command::new("xcrun")
        .args(&symbols_args)
        .status()
        .expect("Failed to extract symbols");

    if !status.success() {
        panic!("Failed to extract symbols");
    }
}

fn setup_rebuild() {
    // Inform cargo to watch all shader files for changes
        // Read all files in the shader directory that end with .metal and save their path
    let shaders_to_check = WalkDir::new(METAL_SHADER_DIR)
        .into_iter()
        .filter_map(|entry| {
            let entry = entry.ok()?; // Ignore errors while traversing
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("metal") {
                Some(path.to_owned())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    // Inform cargo to watch all shader files for changes
    for shader_path in &shaders_to_check {
        if let Some(path_str) = shader_path.to_str() {
            println!("cargo:rerun-if-changed={}", path_str);
            eprintln!("file: {}", path_str);
        } else {
            eprintln!("Warning: Failed to convert shader path to string: {:?}", shader_path);
        }
    }

    // If the build script changes, rebuild the project
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "macos")]
fn get_sdk() -> &'static str {
    if cfg!(feature = "macos") {
        "macosx"
    } else if cfg!(feature = "ios") {
        "ios"
    } else {
        panic!("one of the features macos or ios needs to be enabled")
    }
}
