"""
H.I.M. Model Demonstration
=========================

Interactive demonstration of the Hierarchical Image Model showing
how the different sectors work together to process visual input
and generate intelligent responses.
"""

import numpy as np
import time
import random
from him_model import HIMModel


def create_test_scenarios():
    """Create various test scenarios for the H.I.M. model."""
    scenarios = [
        {
            "name": "Normal Desktop Scene",
            "description": "A typical desktop environment with moderate complexity",
            "image": np.random.rand(480, 640, 3) * 0.6,
            "mouse": {"x": 320, "y": 240},
            "text": "Desktop environment loaded. All systems operational."
        },
        {
            "name": "High Alert Scene",
            "description": "Bright, attention-grabbing scene requiring immediate focus",
            "image": np.ones((480, 640, 3)) * 0.95,
            "mouse": {"x": 50, "y": 50},
            "text": "WARNING: Critical system alert detected! Immediate attention required!"
        },
        {
            "name": "Creative Workspace",
            "description": "Dark, artistic environment stimulating creativity",
            "image": np.random.rand(480, 640, 3) * 0.2,
            "mouse": {"x": 400, "y": 300},
            "text": "Creative mode activated. Generate innovative content and artistic expressions."
        },
        {
            "name": "Social Interaction",
            "description": "Scene suggesting social interaction and bonding",
            "image": np.random.rand(480, 640, 3) * 0.7,
            "mouse": {"x": 200, "y": 180},
            "text": "Social interface active. User interaction detected. Building connections."
        },
        {
            "name": "Threat Assessment",
            "description": "Scene that might trigger self-preservation instincts",
            "image": np.random.rand(480, 640, 3) * 0.1,
            "mouse": {"x": 600, "y": 400},
            "text": "Security scan in progress. Potential threat indicators detected."
        }
    ]
    return scenarios


def run_interactive_demo():
    """Run an interactive demonstration of the H.I.M. model."""
    print("H.I.M. Model Interactive Demonstration")
    print("=" * 50)
    print("This demo shows how the Hierarchical Image Model processes")
    print("visual input through its various sectors and generates")
    print("intelligent responses based on emotions, reasoning, memory,")
    print("and biological drives.")
    print()
    
    # Initialize the model
    him_model = HIMModel()
    
    # Get test scenarios
    scenarios = create_test_scenarios()
    
    while True:
        print("\n" + "=" * 60)
        print("Available Test Scenarios:")
        print("=" * 60)
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"{i}. {scenario['name']}")
            print(f"   {scenario['description']}")
            print()
        
        print("0. Exit Demo")
        print("r. Run Random Scenario")
        print("s. Show System Status")
        print("m. Memory Recall Test")
        print("d. Dream Generation Test")
        
        choice = input("\nSelect an option (0-5, r, s, m, d): ").strip().lower()
        
        if choice == '0':
            print("Thank you for exploring the H.I.M. Model!")
            break
        
        elif choice == 'r':
            scenario = random.choice(scenarios)
            print(f"\nRunning Random Scenario: {scenario['name']}")
            run_scenario(him_model, scenario)
        
        elif choice == 's':
            show_system_status(him_model)
        
        elif choice == 'm':
            test_memory_recall(him_model)
        
        elif choice == 'd':
            test_dream_generation(him_model)
        
        elif choice.isdigit() and 1 <= int(choice) <= len(scenarios):
            scenario = scenarios[int(choice) - 1]
            print(f"\nRunning Scenario: {scenario['name']}")
            run_scenario(him_model, scenario)
        
        else:
            print("Invalid choice. Please try again.")


def run_scenario(him_model, scenario):
    """Run a specific scenario through the H.I.M. model."""
    print(f"\nScenario: {scenario['name']}")
    print(f"Description: {scenario['description']}")
    print(f"Image: {scenario['image'].shape} array, mean brightness: {np.mean(scenario['image']):.3f}")
    print(f"Mouse: {scenario['mouse']}")
    print(f"Text: '{scenario['text']}'")
    print("\n" + "Processing through H.I.M. Model...")
    print("-" * 50)
    
    # Run the cycle
    result = him_model.run_cycle(
        scenario['image'],
        scenario['mouse'],
        scenario['text']
    )
    
    # Display results
    print(f"\nCycle Results:")
    print(f"   Cycle Number: {result['cycle']}")
    print(f"   System State: {result['system_state']}")
    print(f"   Model Decision: {result['model_decision']}")
    print(f"   Control Result: {result['control_result']['status']}")
    print(f"   Dumb Model Result: {result['dumb_model_result']['status']}")
    
    input("\nPress Enter to continue...")


def show_system_status(him_model):
    """Display current system status."""
    print("\nH.I.M. Model System Status")
    print("=" * 40)
    
    status = him_model.get_system_status()
    
    print(f"Cycles Completed: {status['cycle_count']}")
    print(f"System State: {status['system_state']}")
    print()
    
    print("Sector One (Emotions):")
    for emotion, level in status['sector_one_emotions'].items():
        bar = "█" * int(level * 20) + "░" * (20 - int(level * 20))
        print(f"   {emotion.capitalize():12} [{bar}] {level:.2f}")
    
    print("\nSector Two (Reasoning):")
    for metric, level in status['sector_two_reasoning'].items():
        bar = "█" * int(level * 20) + "░" * (20 - int(level * 20))
        print(f"   {metric.replace('_', ' ').title():12} [{bar}] {level:.2f}")
    
    print("\nSector Four (Hormones):")
    for hormone, level in status['sector_four_hormones'].items():
        bar = "█" * int(level * 20) + "░" * (20 - int(level * 20))
        print(f"   {hormone.capitalize():12} [{bar}] {level:.2f}")
    
    print("\nSector Four (Drives):")
    for drive, level in status['sector_four_drives'].items():
        bar = "█" * int(level * 20) + "░" * (20 - int(level * 20))
        print(f"   {drive.replace('_', ' ').title():12} [{bar}] {level:.2f}")
    
    print(f"\nMemory Count: {status['memory_count']}")
    print(f"Dream Fragments: {status['dream_fragments']}")
    print(f"Active Tools: {status['active_tools']}")
    
    input("\nPress Enter to continue...")


def test_memory_recall(him_model):
    """Test memory recall functionality."""
    print("\nMemory Recall Test")
    print("=" * 30)
    
    if not him_model.sector_three.memories:
        print("No memories stored yet. Run some scenarios first!")
        input("\nPress Enter to continue...")
        return
    
    print("Available memory recall tests:")
    print("1. Recall by keyword")
    print("2. Random memory recall")
    print("3. Show all memories")
    
    choice = input("\nSelect test (1-3): ").strip()
    
    if choice == '1':
        keyword = input("Enter keyword to search for: ").strip()
        memory = him_model.sector_three.recall_memory(keyword)
        if memory:
            print(f"Found memory: {str(memory)[:100]}...")
        else:
            print("No memory found with that keyword.")
    
    elif choice == '2':
        memory = him_model.sector_three.recall_memory()
        if memory:
            print(f"Random memory: {str(memory)[:100]}...")
        else:
            print("No memories available.")
    
    elif choice == '3':
        print(f"\nAll Memories ({len(him_model.sector_three.memories)} total):")
        for i, memory in enumerate(him_model.sector_three.memories, 1):
            print(f"{i:2d}. {str(memory['data'])[:80]}...")
            print(f"    Importance: {memory['importance']:.2f}, Access: {memory['access_count']}")
    
    else:
        print("Invalid choice.")
    
    input("\nPress Enter to continue...")


def test_dream_generation(him_model):
    """Test dream generation functionality."""
    print("\nDream Generation Test")
    print("=" * 30)
    
    if not him_model.sector_three.memories:
        print("No memories available for dream generation.")
        print("   Dreams are created from stored memories.")
        input("\nPress Enter to continue...")
        return
    
    print("Generating dream fragment from memories...")
    print("Processing...")
    
    dream = him_model.sector_three.generate_dream_fragment()
    
    if dream:
        print(f"Dream Generated:")
        print(f"   {dream}")
        print(f"\nDream Statistics:")
        print(f"   Total Dreams: {len(him_model.sector_three.dream_fragments)}")
        print(f"   Source Memories: {len(him_model.sector_three.memories)}")
    else:
        print("Dream generation failed.")
    
    input("\nPress Enter to continue...")


def run_automated_demo():
    """Run an automated demonstration of multiple scenarios."""
    print("H.I.M. Model Automated Demonstration")
    print("=" * 50)
    print("Running through all test scenarios automatically...")
    print()
    
    him_model = HIMModel()
    scenarios = create_test_scenarios()
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}/{len(scenarios)}: {scenario['name']}")
        print(f"{scenario['description']}")
        
        result = him_model.run_cycle(
            scenario['image'],
            scenario['mouse'],
            scenario['text']
        )
        
        print(f"Completed - Decision: {result['model_decision']}")
        time.sleep(1)  # Brief pause between scenarios
    
    # Final status
    print(f"\nAutomated Demo Complete!")
    print(f"Final Status:")
    status = him_model.get_system_status()
    print(f"   Cycles: {status['cycle_count']}")
    print(f"   Memories: {status['memory_count']}")
    print(f"   Dreams: {status['dream_fragments']}")
    print(f"   System State: {status['system_state']}")


if __name__ == "__main__":
    print("H.I.M. Model Demonstration")
    print("=" * 40)
    print("Choose demonstration mode:")
    print("1. Interactive Demo (recommended)")
    print("2. Automated Demo")
    
    choice = input("\nSelect mode (1-2): ").strip()
    
    if choice == '1':
        run_interactive_demo()
    elif choice == '2':
        run_automated_demo()
    else:
        print("Invalid choice. Running interactive demo...")
        run_interactive_demo()
