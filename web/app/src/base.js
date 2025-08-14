/**
 * Base Configuration Module
 * 
 * This module provides the foundational configuration for the Snake game,
 * including screen dimensions, colors, game mechanics, and directional mappings.
 * All game components use this base class to ensure consistent configuration.
 */

export class Base {
    /**
     * Base configuration class for the Snake game.
     * 
     * This class centralizes all game configuration including screen dimensions,
     * colors, game mechanics parameters, and directional mappings. It ensures
     * consistent configuration across all game components.
     */

    constructor() {
        // Screen and display configuration
        this._setupScreenConfig();

        // Color palette definition
        this._setupColors();

        // Game mechanics configuration
        this._setupGameConfig();

        // Movement and direction mappings
        this._setupDirectionMappings();
    }

    _setupScreenConfig() {
        /**Configure screen dimensions and related calculations.*/
        this.SCREEN_SIZE = 600;
        this.BLOCK_WIDTH = 20;

        // Calculate maximum valid food position index
        // This ensures food spawns within screen boundaries
        this.MAX_FOOD_INDEX = Math.floor((this.SCREEN_SIZE - this.BLOCK_WIDTH) / this.BLOCK_WIDTH);
    }

    _setupColors() {
        /**Define the color palette used throughout the game.*/
        // Primary game colors (RGB strings for canvas)
        this.BLACK = '#000000';      // Background
        this.WHITE = '#ffffff';      // Text/UI elements
        this.GREEN = '#00ff00';      // Snake body
        this.RED = '#ff0000';        // Apple/food
        this.GRAY = '#c8c8c8';       // Score text
    }

    _setupGameConfig() {
        /**Configure game mechanics and timing parameters.*/
        // Game timing configuration
        this.GAME_SPEED = 10; // Frames per second

        // Enhanced reward system for AI training
        this.REWARDS = {
            'step': -0.02,              // Increased urgency (was -0.01)
            'food': 10,                 // Keep food reward  
            'collision': -15,           // More penalty (was -10)
            'closer_to_food': 0.3,      // Stronger guidance (was 0.1)
            'farther_from_food': -0.3,  // Stronger penalty (was -0.1)
            'survival_bonus': 0.01,     // Doubled (was 0.005)
            'length_bonus': 0.5,        // Keep same
            'inefficient_movement': -0.05  // Keep same
        };
    }

    _setupDirectionMappings() {
        /**
         * Configure directional movement mappings.
         * 
         * Maps direction names to [x, y] coordinate changes.
         * These values represent pixel offsets based on BLOCK_WIDTH.
         */
        this.DIRECTIONS = {
            'right': [this.BLOCK_WIDTH, 0],   // Move right by one block
            'left': [-this.BLOCK_WIDTH, 0],   // Move left by one block
            'up': [0, -this.BLOCK_WIDTH],     // Move up by one block (negative Y)
            'down': [0, this.BLOCK_WIDTH]     // Move down by one block
        };
    }
}
