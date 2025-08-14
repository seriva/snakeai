/**
 * Core Snake game engine - contains all game logic without any rendering dependencies.
 * This is used by both the canvas GUI version and potential future AI training version.
 */

import { Base } from './base.js';

export class SnakeEngine extends Base {
    /**Core Snake game engine - pure game logic without rendering*/

    constructor() {
        super();
        this.reset();
    }

    reset() {
        /**Reset game to initial state*/
        this.snakePositions = [[0, this.BLOCK_WIDTH]];
        this.snakeDirection = 'right';
        this.applePosition = [this.BLOCK_WIDTH * 4, this.BLOCK_WIDTH * 5];
        this.score = 0;
        this.gameOver = false;
        this.reward = 0;
    }

    // Snake properties
    get snakeHead() {
        /**Get snake head position*/
        return this.snakePositions[0];
    }

    get snakeLength() {
        /**Get snake length*/
        return this.snakePositions.length;
    }

    // Direction handling
    setDirection(direction) {
        /**Set snake direction with validation to prevent reverse movement*/
        const opposite = { 'right': 'left', 'left': 'right', 'up': 'down', 'down': 'up' };
        const isOpposite = direction === opposite[this.snakeDirection];

        console.log(`🎯 setDirection: Requested=${direction}, Current=${this.snakeDirection}, Opposite=${opposite[this.snakeDirection]}, Blocked=${isOpposite}`);

        if (!isOpposite) {
            this.snakeDirection = direction;
            console.log(`✅ Direction changed to: ${this.snakeDirection}`);
        } else {
            console.log(`❌ Direction change blocked (reverse movement): ${direction}`);
        }
    }

    // Core game mechanics
    _getNextHeadPosition() {
        /**Get the next head position based on current direction*/
        const [headX, headY] = this.snakePositions[0];
        const [dx, dy] = this.DIRECTIONS[this.snakeDirection];
        return [headX + dx, headY + dy];
    }

    _moveSnake() {
        /**Move snake in current direction*/
        const nextHead = this._getNextHeadPosition();
        // Add new head and always remove tail (growth happens separately)
        this.snakePositions.unshift(nextHead);
        this.snakePositions.pop();
    }

    _growSnake() {
        /**Grow snake by adding segment at tail position*/
        const tail = this.snakePositions[this.snakePositions.length - 1];
        this.snakePositions.push([...tail]); // Create a copy of tail position
        console.log(`🐍 Snake grew! Length: ${this.snakePositions.length}, Tail added at: [${tail[0]}, ${tail[1]}]`);
        console.log(`🐍 Full snake body: ${this.snakePositions.map(p => `[${p[0]},${p[1]}]`).join(' → ')}`);
    }

    _growSnakeWithTail(originalTail) {
        /**Grow snake by adding the original tail position (before movement)*/
        this.snakePositions.push([...originalTail]); // Add back the original tail
        console.log(`🐍 Snake grew! Length: ${this.snakePositions.length}, Original tail added back at: [${originalTail[0]}, ${originalTail[1]}]`);
        console.log(`🐍 Full snake body: ${this.snakePositions.map(p => `[${p[0]},${p[1]}]`).join(' → ')}`);
    }

    _checkWallCollision(position) {
        /**Check if position hits a wall*/
        const [x, y] = position;
        const collision = (x < 0 || x >= this.SCREEN_SIZE || y < 0 || y >= this.SCREEN_SIZE);
        if (collision) {
            console.log(`🧱 Wall collision at [${x},${y}], screen size: ${this.SCREEN_SIZE}`);
        }
        return collision;
    }

    _checkSelfCollision(position) {
        /**Check if position hits snake body*/
        return this.snakePositions.slice(1).some(pos =>
            pos[0] === position[0] && pos[1] === position[1]
        );
    }

    _checkCollision(position) {
        /**Check if position collides with walls or snake body*/
        return this._checkWallCollision(position) || this._checkSelfCollision(position);
    }

    _checkFoodCollision(position) {
        /**Check if position collides with apple*/
        return position[0] === this.applePosition[0] && position[1] === this.applePosition[1];
    }

    _randomizeApplePosition() {
        /**Generate new apple position avoiding snake body*/
        const snakePositionsSet = new Set(
            this.snakePositions.map(pos => `${pos[0]},${pos[1]}`)
        );

        while (true) {
            const x = Math.floor(Math.random() * (this.MAX_FOOD_INDEX + 1)) * this.BLOCK_WIDTH;
            const y = Math.floor(Math.random() * (this.MAX_FOOD_INDEX + 1)) * this.BLOCK_WIDTH;
            const newPos = [x, y];
            const posKey = `${x},${y}`;

            if (!snakePositionsSet.has(posKey)) {
                this.applePosition = newPos;
                break;
            }
        }
    }

    // Main game step
    step(direction = null) {
        /**Execute one game step*/
        // Set direction if provided
        if (direction) {
            this.setDirection(direction);
        }

        // Save the tail position BEFORE moving (in case we need to grow)
        const originalTail = [...this.snakePositions[this.snakePositions.length - 1]];

        // Move snake first (like original game)
        this._moveSnake();
        const headPos = this.snakePositions[0];

        // Set default step reward
        this.reward = this.REWARDS['step'];

        // Check collisions FIRST (before growing)
        const collision = this._checkCollision(headPos);
        console.log(`🎮 Game step: Head at [${headPos[0]},${headPos[1]}], Collision: ${collision}, GameOver: ${this.gameOver}`);

        if (collision) {
            this.gameOver = true;
            this.reward = this.REWARDS['collision'];
            console.log(`💥 GAME OVER! Snake hit obstacle at [${headPos[0]},${headPos[1]}]`);
            return [this.reward, this.gameOver, this.score];
        }

        // Check food collision (only if not game over)
        if (this._checkFoodCollision(headPos)) {
            this.score += 1;
            this._growSnakeWithTail(originalTail);
            this._randomizeApplePosition();
            this.reward = this.REWARDS['food'];
        }

        return [this.reward, this.gameOver, this.score];
    }

    // AI Training utilities
    stepWithAction(action) {
        /**Execute one game step with AI action array*/
        const actionMap = ['right', 'down', 'left', 'up'];
        const directionIdx = action.indexOf(Math.max(...action));
        const direction = actionMap[directionIdx];
        return this.step(direction);
    }

    getStateArray() {
        /**Get current game state as array for AI*/
        const gridSize = Math.floor(this.SCREEN_SIZE / this.BLOCK_WIDTH);
        const state = Array(gridSize).fill().map(() =>
            Array(gridSize).fill().map(() => [0, 0, 0])
        ); // 3 channels: snake head, snake body, apple

        // Mark snake positions
        this.snakePositions.forEach((pos, i) => {
            const x = Math.floor(pos[0] / this.BLOCK_WIDTH);
            const y = Math.floor(pos[1] / this.BLOCK_WIDTH);
            if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
                if (i === 0) { // Head
                    state[y][x][0] = 1;
                } else { // Body
                    state[y][x][1] = 1;
                }
            }
        });

        // Mark apple position
        const appleX = Math.floor(this.applePosition[0] / this.BLOCK_WIDTH);
        const appleY = Math.floor(this.applePosition[1] / this.BLOCK_WIDTH);
        if (appleX >= 0 && appleX < gridSize && appleY >= 0 && appleY < gridSize) {
            state[appleY][appleX][2] = 1;
        }

        return state;
    }

    getObservation() {
        /**Get detailed observation for AI training*/
        const [headX, headY] = this.snakePositions[0];
        const [appleX, appleY] = this.applePosition;

        return {
            snakeHead: [Math.floor(headX / this.BLOCK_WIDTH), Math.floor(headY / this.BLOCK_WIDTH)],
            snakeBody: this.snakePositions.slice(1).map(pos =>
                [Math.floor(pos[0] / this.BLOCK_WIDTH), Math.floor(pos[1] / this.BLOCK_WIDTH)]
            ),
            apple: [Math.floor(appleX / this.BLOCK_WIDTH), Math.floor(appleY / this.BLOCK_WIDTH)],
            direction: this.snakeDirection,
            score: this.score,
            snakeLength: this.snakePositions.length,
            gameOver: this.gameOver
        };
    }
}
