import os
import subprocess


if __name__ == '__main__':
    onnx_path = r"/home/std/workspace-hub/lgc/rf-detr/models/rfdetr-seg-preview.onnx"
    engine_path = r"/home/std/workspace-hub/lgc/rf-detr/models/rfdetr-seg-preview.engine"

    # tensorRT的库路径
    trtexec = r"/home/std/workspace-hub/lgc/TensorRT-10.16.1.11/bin/trtexec"
    cmd = [trtexec,f"--onnx={onnx_path}",f"--saveEngine={engine_path}","--fp16"]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/home/std/workspace-hub/lgc/TensorRT-10.16.1.11/lib:" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(
        cmd,
        text=True,
        timeout=600,
        env=env,
    )

    print("returncode:", result.returncode)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    
