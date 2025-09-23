"""Test random actions in the SMB2 environment."""

import logging
import time

import pytest

from smb2_gym import SuperMarioBros2Env


@pytest.mark.slow
def test_frame_methods_fps_comparison(basic_env_config, caplog):
    """Test and compare FPS for all frame rendering methods with and without human rendering."""

    frame_methods = ["rgb", "grayscale"]
    render_modes = [("human", "Human Render"), (None, "No Render")]
    test_steps = 1000
    all_results = {}

    for render_mode, render_label in render_modes:
        all_results[render_label] = {}

        for method in frame_methods:
            env = SuperMarioBros2Env(
                init_config=basic_env_config,
                render_mode=render_mode,
                action_type="simple",
                frame_method=method,
                env_name="test",
            )

            try:
                start_time = time.time()
                obs, info = env.reset()

                for _ in range(test_steps):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)

                    if terminated or truncated:
                        obs, info = env.reset()

                elapsed_time = time.time() - start_time
                fps = test_steps / elapsed_time

                all_results[render_label][method] = {
                    'fps': fps,
                    'time': elapsed_time,
                    'obs_shape': obs.shape,
                    'obs_dtype': obs.dtype,
                    'memory_kb': obs.nbytes / 1024
                }

            finally:
                env.close()

    # Log comprehensive comparison table
    logger = logging.getLogger(__name__)
    with caplog.at_level(logging.INFO):
        logger.info("Frame Methods FPS Comparison - Complete Results:")
        logger.info("=" * 80)
        logger.info(
            f"{'Method':<12} {'Mode':<12} {'FPS':<8} {'Time(s)':<8} {'Memory(KB)':<12} {'Shape'}"
        )
        logger.info("-" * 80)

        for render_label in ["Human Render", "No Render"]:
            for method in frame_methods:
                data = all_results[render_label][method]
                logger.info(
                    f"{method:<12} {render_label:<12} {data['fps']:<8.2f} "
                    f"{data['time']:<8.2f} {data['memory_kb']:<12.1f} {str(data['obs_shape'])}"
                )

    # Print for visibility
    print(f"\nFrame Methods FPS Comparison - Complete Results:")
    print("=" * 80)
    print(f"{'Method':<12} {'Mode':<12} {'FPS':<8} {'Time(s)':<8} {'Memory(KB)':<12} {'Shape'}")
    print("-" * 80)

    for render_label in ["Human Render", "No Render"]:
        for method in frame_methods:
            data = all_results[render_label][method]
            print(
                f"{method:<12} {render_label:<12} {data['fps']:<8.2f} "
                f"{data['time']:<8.2f} {data['memory_kb']:<12.1f} {str(data['obs_shape'])}"
            )

    # Basic assertions for both render modes
    for render_label, results in all_results.items():
        for method, data in results.items():
            assert data['fps'] > 0, f"{method} FPS should be positive for {render_label}"

            if method == "grayscale":
                assert data['obs_shape'] == (240, 256), f"Grayscale should be 2D for {render_label}"
            else:
                assert data['obs_shape'] == (
                    240, 256, 3
                ), f"RGB methods should be 3D for {render_label}"

        # Performance assertions within each render mode
        # Grayscale should be faster due to smaller memory footprint (60KB vs 180KB)
        assert results['grayscale']['fps'] > results['rgb'][
            'fps'], f"Grayscale should be faster than RGB for {render_label}"

        # RGB performance should be reasonable
        assert results['rgb']['fps'] > 50, f"RGB FPS should be reasonable for {render_label}"

    # Cross-mode performance assertions
    for method in frame_methods:
        no_render_fps = all_results["No Render"][method]['fps']
        human_render_fps = all_results["Human Render"][method]['fps']
        assert no_render_fps > human_render_fps, f"No render should be faster than human render for {method}"

        # Verify memory usage differences
        no_render_mem = all_results["No Render"][method]['memory_kb']
        human_render_mem = all_results["Human Render"][method]['memory_kb']
        assert no_render_mem == human_render_mem, f"Memory usage should be same regardless of render mode for {method}"

    # Memory usage assertions
    for render_label, results in all_results.items():
        assert results['grayscale']['memory_kb'] < results['rgb'][
            'memory_kb'], f"Grayscale should use less memory than RGB for {render_label}"
