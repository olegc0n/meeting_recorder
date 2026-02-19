"""
Utility functions for audio device enumeration and loopback matching.
"""
import soundcard as sc
from typing import List, Dict, Optional, Tuple


def get_output_devices() -> List[Dict[str, str]]:
    """
    Get all available output devices (speakers).
    
    Returns:
        List of dicts with 'name' and 'id' keys
    """
    devices = []
    try:
        speakers = sc.all_speakers()
        for speaker in speakers:
            devices.append({
                'name': speaker.name,
                'id': speaker.id
            })
    except Exception as e:
        print(f"Error enumerating speakers: {e}")
    return devices


def get_loopback_microphones() -> List[Dict[str, str]]:
    """
    Get all available loopback microphones.
    
    Returns:
        List of dicts with 'name' and 'id' keys
    """
    devices = []
    try:
        mics = sc.all_microphones(include_loopback=True)
        for mic in mics:
            # Check if device is loopback (use hasattr for compatibility)
            is_loopback = False
            if hasattr(mic, 'isloopback'):
                is_loopback = mic.isloopback
            else:
                # Fallback: check name for common loopback indicators
                name_lower = mic.name.lower()
                is_loopback = any(keyword in name_lower for keyword in 
                                 ['loopback', 'stereo mix', 'what u hear', 'wave out mix'])
            
            if is_loopback:
                devices.append({
                    'name': mic.name,
                    'id': mic.id
                })
    except Exception as e:
        print(f"Error enumerating loopback microphones: {e}")
    return devices


def find_loopback_for_speaker(speaker_name: str, speaker_id: str) -> Optional[Tuple[str, str]]:
    """
    Find the corresponding loopback microphone for a given speaker.
    
    Args:
        speaker_name: Name of the speaker device
        speaker_id: ID of the speaker device
        
    Returns:
        Tuple of (loopback_name, loopback_id) or None if not found
    """
    loopbacks = get_loopback_microphones()
    
    # Strategy 1: Try to match by similar name patterns
    # Common patterns: "Headphones" -> "Headphones (Loopback)"
    speaker_lower = speaker_name.lower()
    
    for loopback in loopbacks:
        loopback_name_lower = loopback['name'].lower()
        
        # Check if loopback name contains speaker name or vice versa
        if speaker_lower in loopback_name_lower or loopback_name_lower in speaker_lower:
            return (loopback['name'], loopback['id'])
        
        # Check for common loopback naming patterns
        if 'loopback' in loopback_name_lower:
            # Extract base name (e.g., "Headphones (Realtek Audio)" -> "Headphones")
            base_speaker = speaker_lower.split('(')[0].strip()
            if base_speaker and base_speaker in loopback_name_lower:
                return (loopback['name'], loopback['id'])
    
    # Strategy 2: If no match found, return the first available loopback
    # (This is a fallback - user may need to manually select)
    if loopbacks:
        return (loopbacks[0]['name'], loopbacks[0]['id'])
    
    return None
