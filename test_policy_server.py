#!/usr/bin/env python
"""Test script for sending requests to policy server with random images.

This script simulates the observation format expected by the policy server
and sends random images to test the policy inference without any robot interaction.

Example usage:
python test_policy_server.py --policy-server-address https://whippet-pet-singularly.ngrok.app --prompt "pick up the red block"
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import numpy as np
import cv2

# --- openpi specific imports ---
from lerobot.common.utils.websocket_policy import websocket_client_policy as _websocket_client_policy
from lerobot.common.envs.widowx_env import WidowXMessageFormat

resolution = 224

def generate_random_image(width: int = 224, height: int = 224) -> np.ndarray:
    """Generate a random image for testing."""
    # Generate random RGB image
    random_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return random_img

def generate_random_state() -> np.ndarray:
    """Generate a random robot state vector."""
    # Generate random 7D state vector (x, y, z, qx, qy, qz, qw)
    state = np.random.randn(7).astype(np.float32)
    # Normalize quaternion part (last 4 elements)
    quat = state[3:7]
    quat = quat / np.linalg.norm(quat)
    state[3:7] = quat
    return state

def format_test_observation(prompt: str, reset: bool = False) -> Dict[str, Any]:
    """Formats a test observation in the structure expected by the policy."""
    obs_for_policy: WidowXMessageFormat = {
        "state": generate_random_state(),
        "prompt": prompt,
        "reset": reset,
        "images": {
            "image_0": generate_random_image(resolution, resolution)
        },
    }
    return obs_for_policy

def test_policy_inference(
    policy_client: _websocket_client_policy.WebsocketClientPolicy,
    prompt: str,
    num_requests: int = 5
) -> None:
    """Test policy inference with random observations."""
    print(f"Testing policy inference with {num_requests} random requests...")
    print(f"Prompt: {prompt}")
    
    for i in range(num_requests):
        print(f"\n--- Request {i+1}/{num_requests} ---")
        
        # Generate random observation
        obs = format_test_observation(prompt, reset=(i %5 ==  0))
        
        # Print observation details
        print(f"State shape: {obs['state'].shape}")
        print(f"State values: {obs['state']}")
        print(f"Images keys: {list(obs['images'].keys())}")
        for img_key, img in obs['images'].items():
            print(f"  {img_key} shape: {img.shape}, dtype: {img.dtype}")
        
        try:
            # Send request to policy server
            inference_start_time = time.time()
            result = policy_client.infer(obs)
            inference_time = time.time() - inference_start_time
            
            # Process result
            action_chunk = np.array(result["actions"])
            print(f"✓ Received action chunk: shape={action_chunk.shape}")
            print(f"  Action range: min={action_chunk.min(0)}, max={action_chunk.max(0)}")
            print(f"  Inference time: {inference_time:.3f}s")
            
            # Print first few actions for inspection
            if action_chunk.shape[0] > 0:
                print(f"  First action: {action_chunk[0]}")
                if action_chunk.shape[0] > 1:
                    print(f"  Second action: {action_chunk[1]}")
            
        except Exception as e:
            print(f"✗ Error during inference: {e}")
            import traceback
            traceback.print_exc()
        
        # Small delay between requests
        time.sleep(0.5)

def main():
    parser = argparse.ArgumentParser(
        description="Test policy server with random images."
    )
    parser.add_argument(
        "--policy-server-address",
        type=str,
        default="https://whippet-pet-singularly.ngrok.app",
        help="Address (host:port) of the policy server.",
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True, 
        help="Task prompt for the policy."
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=5,
        help="Number of test requests to send."
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Size of the random test images (width=height)."
    )
    
    args = parser.parse_args()
    
    # Update global resolution
    global resolution
    resolution = args.image_size
    
    print(f"Policy server address: {args.policy_server_address}")
    print(f"Image resolution: {resolution}x{resolution}")
    print(f"Number of test requests: {args.num_requests}")
    
    # Initialize policy client
    print(f"\nConnecting to policy server...")
    try:
        policy_client = _websocket_client_policy.WebsocketClientPolicy(
            address=args.policy_server_address
        )
        print("✓ Policy client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize policy client: {e}")
        return
    
    # Run test inference
    try:
        test_policy_inference(policy_client, args.prompt, args.num_requests)
        print(f"\n✓ Completed {args.num_requests} test requests successfully")
    except KeyboardInterrupt:
        print(f"\n⚠ Test interrupted by user")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
