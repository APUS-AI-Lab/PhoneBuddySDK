//! Generates phone_buddy.h via cbindgen when the BUILD_HEADER env var is
//! set (used by scripts/gen-bindings.sh). Keeping it opt-in avoids a
//! cbindgen run on every cargo build.

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "android" {
        cc::Build::new()
            .file("c_src/phonebuddy_jni.c")
            .include("include")
            .warnings(false)
            .compile("phonebuddy_jni");
        println!("cargo::rerun-if-changed=c_src/phonebuddy_jni.c");
    }

    if std::env::var("CARGO_CFG_TARGET_OS").is_ok() && std::env::var("PB_BUILD_HEADER").is_ok() {
        let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let out = std::path::Path::new(&crate_dir).join("include").join("phone_buddy.h");
        std::fs::create_dir_all(out.parent().unwrap()).ok();
        let config = cbindgen::Config::from_root_or_default(&crate_dir);
        cbindgen::Builder::new()
            .with_crate(&crate_dir)
            .with_config(config)
            .generate()
            .expect("cbindgen failed")
            .write_to_file(out);
    }
    println!("cargo::rerun-if-env-changed=PB_BUILD_HEADER");
    println!("cargo::rerun-if-changed=cbindgen.toml");
}


