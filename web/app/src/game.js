/**
 * Snake Game - Canvas GUI Implementation
 *
 * This module provides a graphical interface for the Snake game using HTML5 Canvas.
 * It uses the SnakeEngine for all game logic and focuses solely on rendering and user input.
 */

import { Base } from './base.js';
import { SnakeEngine } from './snakeEngine.js';

class SnakeRenderer {
    /**
     * Handles rendering of the snake using canvas.
     * 
     * This class is responsible for drawing the snake segments on the screen
     * using efficient canvas rectangle operations.
     */

    constructor(ctx, baseConfig) {
        /**
         * Initialize the snake renderer.
         * 
         * Args:
         *     ctx: The canvas 2D rendering context
         *     baseConfig: Base configuration object containing colors and dimensions
         */
        this.ctx = ctx;
        this.color = baseConfig.GREEN;
        this.blockWidth = baseConfig.BLOCK_WIDTH;
    }

    draw(snakePositions) {
        /**
         * Draw snake using canvas rectangles for optimal performance.
         * 
         * Args:
         *     snakePositions: Array of [x, y] coordinates for each snake segment
         */
        this.ctx.fillStyle = this.color;
        snakePositions.forEach(pos => {
            this.ctx.fillRect(pos[0], pos[1], this.blockWidth, this.blockWidth);
        });
    }
}

class AppleRenderer {
    /**
     * Handles rendering of the apple using canvas.
     * 
     * This class is responsible for drawing the apple/food item on the screen
     * using efficient canvas rectangle operations.
     */

    constructor(ctx, baseConfig) {
        /**
         * Initialize the apple renderer.
         * 
         * Args:
         *     ctx: The canvas 2D rendering context
         *     baseConfig: Base configuration object containing colors and dimensions
         */
        this.ctx = ctx;
        this.color = baseConfig.RED;
        this.blockWidth = baseConfig.BLOCK_WIDTH;
    }

    draw(applePosition) {
        /**
         * Draw apple using canvas rectangle for optimal performance.
         * 
         * Args:
         *     applePosition: [x, y] coordinates of the apple
         */
        this.ctx.fillStyle = this.color;
        this.ctx.fillRect(applePosition[0], applePosition[1], this.blockWidth, this.blockWidth);
    }
}

export class Game extends Base {
    /**
     * Main Game class that handles the Canvas GUI for Snake.
     * 
     * This class manages the game canvas, user input, rendering, and coordinates
     * with the SnakeEngine for all game logic. Provides manual gameplay interface.
     */

    // UI Constants
    static SCORE_FONT_SIZE = '20px';
    static GAME_OVER_FONT_SIZE = '30px';
    static SCORE_POSITION_OFFSET = 120;
    static SCORE_MARGIN = 10;
    static GAME_OVER_TEXT_OFFSET = 40;

    constructor(canvas) {
        /**Initialize the game with canvas setup and create necessary components.*/
        super();

        // Initialize canvas
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');

        // Set canvas size
        this.canvas.width = this.SCREEN_SIZE;
        this.canvas.height = this.SCREEN_SIZE;

        // Use the core engine for all game logic
        this.engine = new SnakeEngine();

        // Create renderers for drawing components
        this.snakeRenderer = new SnakeRenderer(this.ctx, this);
        this.appleRenderer = new AppleRenderer(this.ctx, this);

        // Game state flags
        this.gameStarted = false;
        this.aiMode = false; // Always false - manual play only


        // Direction mapping for keyboard input
        this.directionMap = {
            'ArrowUp': 'up',
            'ArrowDown': 'down',
            'ArrowLeft': 'left',
            'ArrowRight': 'right'
        };

        // Bind event handlers
        this._setupEventHandlers();

        // Start game loop
        this._startGameLoop();
    }

    _setupEventHandlers() {
        /**Setup keyboard event handlers*/
        document.addEventListener('keydown', (event) => {
            this.handleInput(event.key);
        });
    }

    resetGame() {
        /**
         * Reset game to initial state.
         * 
         * This delegates to the engine's reset method to maintain separation of concerns.
         */
        this.engine.reset();
        // Reset game started flag
        this.gameStarted = false;
    }

    setAIMode(enabled) {
        /**
         * Legacy method - AI mode is always disabled.
         * Kept for compatibility but does nothing.
         */
        this.aiMode = false; // Always manual mode
        console.log('Manual mode - AI functionality removed');
    }

    updateGameState() {
        /**
         * Update game state for one frame.
         * 
         * All game logic is handled by the engine to maintain clean architecture.
         */
        this.engine.step();
    }

    draw() {
        /**
         * Draw the entire game state to the screen.
         * 
         * This method handles the start screen, active game state, and game over screen.
         */
        this.ctx.fillStyle = this.BLACK;
        this.ctx.fillRect(0, 0, this.SCREEN_SIZE, this.SCREEN_SIZE);

        if (!this.gameStarted) {
            this._drawStartScreen();
        } else if (!this.engine.gameOver) {
            this._drawActiveGame();
        } else {
            this._drawGameOverScreen();
        }
    }

    _drawActiveGame() {
        /**Draw the active game state including snake, apple, and score.*/
        this.snakeRenderer.draw(this.engine.snakePositions);
        this.appleRenderer.draw(this.engine.applePosition);
        this._drawScore();
    }

    _drawScore() {
        /**Draw the current score in the top-right corner.*/
        this.ctx.fillStyle = this.GRAY;
        this.ctx.font = `${Game.SCORE_FONT_SIZE} Arial`;
        const scoreText = `Score: ${this.engine.score}`;
        const scoreX = this.SCREEN_SIZE - Game.SCORE_POSITION_OFFSET;
        this.ctx.fillText(scoreText, scoreX, Game.SCORE_MARGIN + 20);
    }

    _drawStartScreen() {
        /**
         * Draw the start screen with game title and start instruction.
         * 
         * Centers the text on the screen for better visual presentation.
         */
        const centerX = this.SCREEN_SIZE / 2;
        const centerY = this.SCREEN_SIZE / 2;

        this.ctx.textAlign = 'center';
        this.ctx.fillStyle = this.WHITE;

        // Game title
        this.ctx.font = `${Game.GAME_OVER_FONT_SIZE} Arial`;
        this.ctx.fillText('Snake Game', centerX, centerY - Game.GAME_OVER_TEXT_OFFSET);

        // Start instruction
        this.ctx.font = `${Game.SCORE_FONT_SIZE} Arial`;
        this.ctx.fillText('Press SPACE to start', centerX, centerY);

        // Reset text alignment
        this.ctx.textAlign = 'left';
    }

    _drawGameOverScreen() {
        /**
         * Draw the game over screen with final score and restart instruction.
         * 
         * Centers the text on the screen for better visual presentation.
         */
        const centerX = this.SCREEN_SIZE / 2;
        const centerY = this.SCREEN_SIZE / 2;

        this.ctx.textAlign = 'center';
        this.ctx.fillStyle = this.WHITE;

        // Game over message
        this.ctx.font = `${Game.GAME_OVER_FONT_SIZE} Arial`;
        this.ctx.fillText('Game Over! Press SPACE to restart', centerX, centerY);

        // Final score
        this.ctx.fillText(`Final Score: ${this.engine.score}`, centerX, centerY + Game.GAME_OVER_TEXT_OFFSET);

        // Reset text alignment
        this.ctx.textAlign = 'left';
    }

    handleInput(key) {
        /**
         * Handle keyboard input for game control.
         * 
         * Supported keys:
         * - Arrow keys: Control snake direction
         * - Space: Start game or restart when game over
         * 
         * Args:
         *     key: string representing the pressed key
         */
        // Manual mode - keyboard input always enabled

        if (key === ' ') {
            if (!this.gameStarted) {
                this.gameStarted = true;
            } else if (this.engine.gameOver) {
                this.resetGame();
                this.gameStarted = true;
            }
        } else if (key in this.directionMap && this.gameStarted) {
            this.engine.setDirection(this.directionMap[key]);
        }
    }

    runAiStep(move) {
        /**
         * Legacy method - AI training is no longer supported.
         * Kept for compatibility but returns default values.
         */
        console.warn('runAIStep called but AI functionality has been removed');
        return [0, this.engine.gameOver, this.engine.score];
    }

    _startGameLoop() {
        /**Start the main game loop*/
        const gameLoop = () => {
            // Update game state only if game has started and is active
            if (this.gameStarted && !this.engine.gameOver) {
                this.updateGameState();
            }

            // Render the current frame
            this.draw();

            // Continue the loop
            setTimeout(() => {
                requestAnimationFrame(gameLoop);
            }, 1000 / this.GAME_SPEED);
        };

        // Start the loop
        requestAnimationFrame(gameLoop);
    }

    // Convenience properties for backward compatibility with existing code
    get score() {
        /**Get current game score.*/
        return this.engine.score;
    }

    get gameOver() {
        /**Get current game over status.*/
        return this.engine.gameOver;
    }

    get reward() {
        /**Get current reward value.*/
        return this.engine.reward;
    }
}


