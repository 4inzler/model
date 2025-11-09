#!/usr/bin/env python3
"""
Quick Luna Training Script
=========================

A faster version of the Luna memory integration for testing purposes.
"""

import sqlite3
import numpy as np
import json
import time
import pickle
import os
from datetime import datetime
from him_model import HIMModel, VectorizedImage

def quick_luna_training():
    """Quick training of H.I.M. model on Luna's memories."""
    print("Starting Quick Luna Training...")
    
    # Initialize H.I.M. model
    him_model = HIMModel()
    
    try:
        # Connect to Luna's database
        conn = sqlite3.connect('luna_memories.db')
        cursor = conn.cursor()
        
        # Get a sample of memories (first 1000 for quick training)
        cursor.execute("SELECT id, timestamp, memory_type, content, mood, importance FROM memories LIMIT 1000")
        memories = cursor.fetchall()
        
        # Get a sample of conversations (first 500 for quick training)
        cursor.execute("SELECT id, timestamp, user_message, luna_response, mood, voice_used FROM conversations LIMIT 500")
        conversations = cursor.fetchall()
        
        conn.close()
        
        print(f"Processing {len(memories)} memories and {len(conversations)} conversations...")
        
        # Process memories
        for i, memory in enumerate(memories):
            if i % 100 == 0:
                print(f"Processing memory {i}/{len(memories)}")
            
            # Create vector from memory
            vector = np.random.rand(10, 10) * 0.8 + 0.1  # Simplified vector creation
            
            # Create vectorized image
            vectorized_img = VectorizedImage(
                data=vector,
                metadata={
                    'thought': memory[3][:50] + "..." if len(memory[3]) > 50 else memory[3],
                    'type': 'luna_memory',
                    'original_type': memory[2],
                    'mood': memory[4],
                    'importance': memory[5],
                    'memory_id': memory[0]
                },
                timestamp=time.time(),
                memory_id=f"luna_memory_{memory[0]}"
            )
            
            # Store in H.I.M.'s spatial memory
            him_model.sector_three.store_spatial_memory(vectorized_img, memory[5] / 5.0)
        
        # Process conversations
        for i, conversation in enumerate(conversations):
            if i % 100 == 0:
                print(f"Processing conversation {i}/{len(conversations)}")
            
            # Create vector from conversation
            vector = np.random.rand(10, 10) * 0.8 + 0.1  # Simplified vector creation
            
            # Create vectorized image
            vectorized_img = VectorizedImage(
                data=vector,
                metadata={
                    'thought': conversation[2][:50] + "..." if len(conversation[2]) > 50 else conversation[2],
                    'type': 'luna_conversation',
                    'mood': conversation[4],
                    'voice_used': conversation[5],
                    'conversation_id': conversation[0]
                },
                timestamp=time.time(),
                memory_id=f"luna_conversation_{conversation[0]}"
            )
            
            # Store in H.I.M.'s spatial memory
            him_model.sector_three.store_spatial_memory(vectorized_img, 3.0 / 5.0)  # Medium importance
        
        # Create training directory
        training_dir = "him_luna_training"
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
        
        # Save trained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = os.path.join(training_dir, f"him_trained_on_luna_{timestamp}.pkl")
        
        with open(model_file, 'wb') as f:
            pickle.dump(him_model, f)
        
        # Save training metadata
        metadata = {
            'training_timestamp': timestamp,
            'luna_memories_processed': len(memories),
            'luna_conversations_processed': len(conversations),
            'him_spatial_memories': len(him_model.sector_three.spatial_memory_map),
            'him_regular_memories': len(him_model.sector_three.memories),
            'training_type': 'quick_training'
        }
        
        metadata_file = os.path.join(training_dir, f"training_metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nQuick Luna Training Complete!")
        print(f"Model saved to: {model_file}")
        print(f"Metadata saved to: {metadata_file}")
        print(f"Total Luna memories integrated: {len(memories) + len(conversations)}")
        print(f"H.I.M. spatial memories: {len(him_model.sector_three.spatial_memory_map)}")
        
        return model_file, metadata_file
        
    except Exception as e:
        print(f"Error during quick training: {e}")
        return None, None

if __name__ == "__main__":
    quick_luna_training()
