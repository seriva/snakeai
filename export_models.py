#!/usr/bin/env python3
"""
Export trained DQN models to ONNX format for JavaScript inference.

This script loads the trained DQN V1 and V2 models and exports them to ONNX format,
which can then be used with ONNX Runtime Web in JavaScript applications.
"""

from dqn_agent_v1 import DQNAgentV1
from dqn_agent_v2 import DQNAgentV2


def export_all_models():
    """Export both DQN V1 and V2 models to ONNX format."""
    print("🚀 Starting model export to ONNX format...")
    
    # Export DQN V1 model
    print("\n📦 Exporting DQN V1 model...")
    agent_v1 = DQNAgentV1()
    if agent_v1.load_model("model/dqn_agent_v1.pth", "model/dqn_agent_v1_data.json"):
        agent_v1.export_to_onnx("web/app/model/dqn_agent_v1.onnx")
    else:
        print("❌ Could not load DQN V1 model - make sure it's trained first!")
    
    # Export DQN V2 model
    print("\n📦 Exporting DQN V2 model...")
    agent_v2 = DQNAgentV2()
    if agent_v2.load_model("model/dqn_agent_v2.pth", "model/dqn_agent_v2_data.json"):
        agent_v2.export_to_onnx("web/app/model/dqn_agent_v2.onnx")
    else:
        print("❌ Could not load DQN V2 model - make sure it's trained first!")
    
    print("\n✨ Export process completed!")
    print("📁 ONNX models saved in the 'web/app/model/' directory")
    print("🌐 You can now use these models in JavaScript with ONNX Runtime Web")


if __name__ == "__main__":
    export_all_models()
