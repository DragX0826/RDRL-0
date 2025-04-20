import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, List
from scipy.ndimage import binary_erosion, binary_dilation

class TerrainGenerator:
    def __init__(self, seed: int = 42):
        print("[DEBUG] TerrainGenerator init")
        """Initialize the terrain generator with a random seed."""
        np.random.seed(seed)
        
    def generate_perlin_noise(self, shape: Tuple[int, int], scale: float = 0.1) -> np.ndarray:
        """Generate 2D Perlin noise for terrain height."""
        x = np.linspace(0, scale, shape[0])
        y = np.linspace(0, scale, shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Generate noise using multiple octaves
        noise = np.zeros(shape)
        for octave in range(4):
            frequency = 2 ** octave
            amplitude = 0.5 ** octave
            noise += amplitude * np.sin(frequency * X) * np.cos(frequency * Y)
        
        return noise
    
    def generate_3d_noise(self, shape: Tuple[int, int, int], scale: float = 0.1) -> np.ndarray:
        """Generate 3D Perlin noise for cave systems."""
        noise = np.zeros(shape)
        for z in range(shape[2]):
            layer = self.generate_perlin_noise(shape[:2], scale=scale + z*0.02)
            noise[:, :, z] = layer
        return noise
    
    def generate_3d_caves(self, size: int = 64, threshold: float = 0.5) -> np.ndarray:
        """Generate 3D cave systems using 3D noise thresholding."""
        noise = self.generate_3d_noise((size, size, size))
        caves = (noise > threshold).astype(float)
        
        # Apply erosion to make caves more natural
        for _ in range(2):
            caves = binary_erosion(caves, structure=np.ones((3,3,3))).astype(caves.dtype)
        for _ in range(1):
            caves = binary_dilation(caves, structure=np.ones((3,3,3))).astype(caves.dtype)
            
        return caves
    
    def generate_biome_map(self, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Generate temperature and humidity maps for biome classification."""
        temperature = self.generate_perlin_noise(shape, scale=0.2)
        humidity = self.generate_perlin_noise(shape, scale=0.2)
        
        # Normalize to [0, 1]
        temperature = (temperature - temperature.min()) / (temperature.max() - temperature.min())
        humidity = (humidity - humidity.min()) / (humidity.max() - humidity.min())
        
        return {
            'temperature': temperature,
            'humidity': humidity
        }
    
    def generate_terrain(self, size: int = 256, difficulty: float = 0.5) -> Dict[str, np.ndarray]:
        """Generate a complete terrain with height map and biome information."""
        shape = (size, size)
        
        # Generate base terrain with difficulty scaling
        base_terrain = self.generate_perlin_noise(shape, scale=0.1)
        
        # Add detail noise with difficulty-based amplitude
        detail_noise = self.generate_perlin_noise(shape, scale=0.5)
        terrain = base_terrain + (0.2 + 0.3 * difficulty) * detail_noise
        
        # Generate biome information
        biome_info = self.generate_biome_map(shape)
        
        # Apply smoothing with difficulty-based intensity
        terrain = gaussian_filter(terrain, sigma=1 + difficulty)
        
        # Generate caves if difficulty is high enough
        caves = None
        if difficulty > 0.7:
            cave_size = int(size * 0.25)  # Scale cave size with terrain
            caves = self.generate_3d_caves(cave_size)
        
        return {
            'height_map': terrain,
            'temperature': biome_info['temperature'],
            'humidity': biome_info['humidity'],
            'caves': caves
        }
    
    def get_biome_type(self, temperature: float, humidity: float) -> str:
        """Determine biome type based on temperature and humidity values."""
        if temperature > 0.7:
            if humidity > 0.5:
                return "jungle"
            else:
                return "desert"
        elif temperature > 0.3:
            if humidity > 0.5:
                return "forest"
            else:
                return "plains"
        else:
            if humidity > 0.5:
                return "taiga"
            else:
                return "tundra"
    
    def get_biome_properties(self, biome_type: str) -> Dict[str, float]:
        """Get physical properties for each biome type."""
        properties = {
            "jungle": {"friction": 0.8, "bounce": 0.1},
            "desert": {"friction": 0.3, "bounce": 0.2},
            "forest": {"friction": 0.6, "bounce": 0.15},
            "plains": {"friction": 0.5, "bounce": 0.1},
            "taiga": {"friction": 0.4, "bounce": 0.2},
            "tundra": {"friction": 0.2, "bounce": 0.3}
        }
        return properties.get(biome_type, {"friction": 0.5, "bounce": 0.1}) 