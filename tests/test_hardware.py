from source2reel.hardware import classify_gpu_vendor, parse_gpu_devices, select_inference_backend


def test_amd_vulkan_detection_selects_radv():
    text = "0d:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Radeon RX 9060 XT"
    devices = parse_gpu_devices(text)
    assert classify_gpu_vendor(devices) == "amd"
    assert select_inference_backend("amd", True) == "llama.cpp+vulkan/radv"


def test_nvidia_is_not_inferred_from_platform():
    assert classify_gpu_vendor(()) == "cpu-only"
    assert select_inference_backend("cpu-only", False) == "llama.cpp+cpu"


def test_nvidia_with_vulkan_still_uses_vulkan_not_cuda():
    text = "01:00.0 VGA compatible controller: NVIDIA Corporation Example GPU"
    devices = parse_gpu_devices(text)
    assert classify_gpu_vendor(devices) == "nvidia"
    assert select_inference_backend("nvidia", True) == "llama.cpp+vulkan"
