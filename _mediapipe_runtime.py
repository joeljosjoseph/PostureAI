from importlib.metadata import PackageNotFoundError, version


def ensure_mediapipe_runtime_compatible() -> None:
    """
    MediaPipe 0.10.x expects protobuf versions that still expose GetPrototype().
    Fail fast with a clear message when a newer protobuf is installed.
    """
    try:
        protobuf_version = version("protobuf")
    except PackageNotFoundError:
        return

    major_text = protobuf_version.split(".", 1)[0]
    try:
        major = int(major_text)
    except ValueError:
        return

    if major >= 5:
        raise RuntimeError(
            "Incompatible protobuf runtime detected for MediaPipe: "
            f"protobuf=={protobuf_version}. "
            "Use protobuf<5 for this project, for example `pip install protobuf==4.25.8`, "
            "or run the project with the intended virtual environment."
        )
