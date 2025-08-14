/**
 * Main entry point for the Snake game (Manual Play Only)
 * 
 * This module initializes the game and starts the main game loop.
 * AI functionality has been removed for simplified manual gameplay.
 */

import { Game } from './game.js';

/**
 * Simple Snake Game Manager - handles manual gameplay only
 */
class SnakeGameDemo {
    constructor() {
        this.game = null;
        this.isRunning = false;
        this.bestScore = 0;

        this.setupUI();
    }

    setupUI() {
        // Get UI elements
        this.elements = {
            gameMode: document.getElementById('gameMode'),
            startButton: document.getElementById('startButton'),
            resetButton: document.getElementById('resetButton'),
            currentScore: document.getElementById('currentScore'),
            bestScore: document.getElementById('bestScore'),
            statusMessage: document.getElementById('statusMessage')
        };

        // Setup event listeners
        this.elements.startButton.addEventListener('click', () => this.toggleGame());
        this.elements.resetButton.addEventListener('click', () => this.resetGame());

        // Remove unused game mode change handler
        this.elements.gameMode.addEventListener('change', () => {
            this.updateStatus('Manual play mode selected');
        });
    }

    toggleGame() {
        if (this.isRunning) {
            this.stopGame();
        } else {
            this.startGame();
        }
    }

    startGame() {
        try {
            this.updateStatus('Starting manual game...', 'loading');

            // Initialize game if not exists
            if (!this.game) {
                const canvas = document.getElementById('gameCanvas');
                this.game = new Game(canvas);
            }

            // Reset and start manual game
            this.game.resetGame();
            this.game.setAIMode(false); // Ensure AI is disabled

            this.isRunning = true;
            this.elements.startButton.textContent = 'Stop Game';

            this.updateStatus('Manual game started - Use arrow keys to play!', 'ready');

            // Start monitoring game status
            this.startGameMonitoring();

        } catch (error) {
            this.updateStatus('Error: ' + error.message, 'error');
            console.error('Game start error:', error);
        }
    }

    startGameMonitoring() {
        const checkGameStatus = () => {
            if (this.isRunning && this.game) {
                // Update score display
                this.updateGameStats();

                // Check if game is over
                if (this.game.gameOver) {
                    this.handleGameOver();
                } else {
                    // Continue monitoring
                    setTimeout(checkGameStatus, 100);
                }
            }
        };

        checkGameStatus();
    }

    handleGameOver() {
        const finalScore = this.game.score;

        // Update best score
        if (finalScore > this.bestScore) {
            this.bestScore = finalScore;
        }

        this.updateGameStats();
        this.stopGame();
        this.updateStatus(`Game Over! Final Score: ${finalScore}`, 'ready');

        console.log(`🏁 Game ended with score: ${finalScore}`);
    }

    stopGame() {
        this.isRunning = false;
        this.elements.startButton.textContent = 'Start Game';

        if (!this.game || !this.game.gameOver) {
            this.updateStatus('Game stopped', 'ready');
        }
    }

    resetGame() {
        this.stopGame();

        if (this.game) {
            this.game.resetGame();
            this.game.setAIMode(false);
        }

        // Reset scores but keep best score
        this.updateGameStats();
        this.updateStatus('Game reset - Ready to play!', 'ready');
    }

    updateGameStats() {
        const currentScore = this.game ? this.game.score : 0;

        this.elements.currentScore.textContent = currentScore;
        this.elements.bestScore.textContent = this.bestScore;
    }

    updateStatus(message, type = '') {
        this.elements.statusMessage.textContent = message;
        this.elements.statusMessage.className = 'status ' + type;
    }
}

/**
 * Main function that initializes the Snake game demo.
 */
function main() {
    try {
        const canvas = document.getElementById('gameCanvas');
        if (!canvas) {
            throw new Error('Canvas element not found');
        }

        // Create simple game demo manager
        const demo = new SnakeGameDemo();

        console.log('Snake game initialized successfully (Manual play only)');

    } catch (error) {
        console.error('Failed to initialize Snake game:', error);

        // Display error message to user
        const errorElement = document.createElement('div');
        errorElement.style.color = 'red';
        errorElement.style.textAlign = 'center';
        errorElement.style.padding = '20px';
        errorElement.innerHTML = `
            <h2>Failed to load Snake Game</h2>
            <p>Error: ${error.message}</p>
            <p>Please refresh the page to try again.</p>
        `;
        document.body.appendChild(errorElement);
    }
}

// Initialize the game when the page loads
window.addEventListener('load', main);