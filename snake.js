class SnakeGame {
    refreshRate = 1000 / 10; //10 frames a second
    speedIndex = 1;
    interval;
    canvas;
    ctx;
    gridX;
    gridY;
    trail = [];
    tail = 5;
    dir = {x: 1, y: 0};
    pos = {x: 0, y: 0};
    sectionDimension = 10;
    sectionOffset = 1;
    pause = false;
    appleCoords = {x: 0, y: 0};
    skipBrain = false;
    learnIteration = 1;
    //rewards
    aliveReward = 0;
    deadReward = 0;
    appleReward = 0;

    constructor() {
        this.canvas = document.getElementById('snake');
        this.speedIndex = document.getElementById('speed').value;
        this.ctx = this.canvas.getContext('2d');
        this.reset();

        if (this.ctx) {
            this.hookListeners();
            this.placeApple();
            this.draw(); // draw first frame to init all positions
            this.initAI();
            this.setSpeed();
        }

    }

    initAI() {
        this.loadBrain();
        var num_inputs = this.gridX * this.gridY; // Grid representing the board
        var num_actions = 4; // 4 possible angles agent can turn
        var temporal_window = 1; // amount of temporal memory. 0 = agent lives in-the-moment :)
        var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

        // the value function network computes a value of taking any of the possible actions
        // given an input state. Here we specify one explicitly the hard way
        // but user could also equivalently instead use opt.hidden_layer_sizes = [20,20]
        // to just insert simple relu hidden layers.
        var layer_defs = [];
        layer_defs.push({type: 'input', out_sx: 1, out_sy: 1, out_depth: network_size});
        layer_defs.push({type: 'fc', num_neurons: 50, activation: 'relu'});
        layer_defs.push({type: 'fc', num_neurons: 50, activation: 'relu'});
        layer_defs.push({type: 'regression', num_neurons: num_actions});

        // options for the Temporal Difference learner that trains the above net
        // by backpropping the temporal difference learning rule.
        var tdtrainer_options = {learning_rate: 0.001, momentum: 0.0, batch_size: 64, l2_decay: 0.01};

        var opt = {};
        opt.temporal_window = temporal_window;
        opt.experience_size = 30000;
        opt.start_learn_threshold = 1000;
        opt.gamma = 0.7;
        opt.learning_steps_total = 200000;
        opt.learning_steps_burnin = 3000;
        opt.epsilon_min = 0.05;
        opt.epsilon_test_time = 0.05;
        opt.layer_defs = layer_defs;
        opt.tdtrainer_options = tdtrainer_options;

        this.brain = new deepqlearn.Brain(num_inputs, num_actions, opt); // woohoo
        if (this.network) {
            this.brain.value_net.fromJSON(this.network);
        }
    }

    draw() {
        if (this.pause)
            return;

        this.clear();

        let currentPos = this.move();

        this.drawApple();

        this.ctx.fillStyle = 'white';
        for (let i = 0; i < this.trail.length; i++) {
            if (i === this.trail.length - 1) {
                this.ctx.fillStyle = '#ffcb00';
            }
            this.ctx.fillRect(this.trail[i].x * this.sectionDimension, this.trail[i].y * this.sectionDimension, this.sectionDimension - this.sectionOffset, this.sectionDimension - this.sectionOffset);
            if (this.trail[i].x == currentPos.x && this.trail[i].y == currentPos.y && i != 0) {
                this.die();
                this.reset()
            }
        }
        this.checkApple();

        //apply reward after all ways to die have been checked.
        this.applyReward();

    }

    hookListeners() {
        document.addEventListener('keydown', function (event) {
            switch (event.which || event.keyCode) {
                case 32:
                    this.pause = !this.pause;
                    break;
                case 37:
                    if (this.dir.x != 1) {
                        this.dir = {x: -1, y: 0};
                        this.pause = false;
                    }
                    break;
                case 38:
                    if (this.dir.y != 1) {
                        this.dir = {x: 0, y: -1};
                        this.pause = false;
                    }
                    break;
                case 39:
                    if (this.dir.x != -1) {
                        this.dir = {x: 1, y: 0};
                        this.pause = false;
                    }

                    break;
                case 40:
                    if (this.dir.y != -1) {
                        this.dir = {x: 0, y: 1};
                        this.pause = false;
                    }

                    break;
                case 107:
                    this.speedIndex++;
                    this.resetSpeed(this.speedIndex);

                    break;
                case 109:
                    this.speedIndex--;
                    this.resetSpeed(this.speedIndex);

                    break;
                case 17:
                    this.skipBrain = !this.skipBrain;

                    break;
                default:
                    console.log(event.which || event.keyCode);
                    break;

            }
        }.bind(this));
        document.getElementById('speed').addEventListener('keyup', this.resetSpeed.bind(this));
        document.getElementById('speed').addEventListener('change', this.resetSpeed.bind(this));
    }

    resetSpeed(factor) {
        if (isNaN(factor)) {
            factor = parseFloat(document.getElementById('speed').value);
        }
        if (!isNaN(factor) && factor > 0) {
            this.speedIndex = factor;
        } else {
            this.speedIndex = 1;
        }
        document.getElementById('speed').value = factor;
        this.setSpeed();
    }

    setSpeed() {
        clearInterval(this.interval);
        this.interval = setInterval(this.draw.bind(this), this.refreshRate / this.speedIndex);

    }

    getCurrentPlayGround() {
        let valueArr = [];
        for (let y = 0; y <= this.gridY; y++) {
            for (let x = 0; x <= this.gridX; x++) {
                valueArr.push(this.intersectValue(x, y));
            }
        }
        return valueArr;
    }

    intersectValue(x, y) {
        // check if the snake is there
        if (this.appleCoords.x == x && this.appleCoords.y == y)
            return 1;

        for (let ind in this.trail) {
            if (this.trail[ind].x == x && this.trail[ind].y == y) {
                return -1;
            }
        }
        return 0;
    }

    applyReward() {
        if (this.brain && !this.skipBrain) {
            console.log('Apply reward', this.aliveReward + this.appleReward + this.deadReward);
            this.brain.backward(this.aliveReward + this.appleReward + this.deadReward);
            this.aliveReward = 0;
            this.appleReward = 0;
            this.deadReward = 0;
        }
    }

    move() {
        let action = 0;
        if (this.brain && !this.skipBrain) {
            let grid = this.getCurrentPlayGround();
            let action = this.brain.forward(grid);
            console.log('Iteration', this.learnIteration);
            switch (action) {
                case 0:
                    if (this.dir.x != 1) {
                        this.dir = {x: -1, y: 0};
                        this.pause = false;
                    }
                    break;
                case 1:
                    if (this.dir.y != 1) {
                        this.dir = {x: 0, y: -1};
                        this.pause = false;
                    }
                    break;
                case 2:
                    if (this.dir.x != -1) {
                        this.dir = {x: 1, y: 0};
                        this.pause = false;
                    }

                    break;
                case 3:
                    if (this.dir.y != -1) {
                        this.dir = {x: 0, y: 1};
                        this.pause = false;
                    }
            }
            this.saveBrain();
        }

        this.pos.x += this.dir.x;
        this.pos.y += this.dir.y;

        if (this.pos.x < 0) {
            this.die();
            this.reset();
        }
        if (this.pos.x > this.gridX) {
            this.die();
            this.reset();
        }
        if (this.pos.y < 0) {
            this.die();
            this.reset();
        }
        if (this.pos.y > this.gridY) {
            this.die();
            this.reset();
        }
        //we are not dead.. YEY !
        this.aliveReward = 1;

        this.trail.push(JSON.parse(JSON.stringify(this.pos)));
        while (this.trail.length > this.tail) {
            this.trail.shift();
        }
        return this.trail[0] || {x: -1, y: -1};
    }

    placeApple() {
        this.appleCoords.x = Math.floor(Math.random() * this.gridX);
        this.appleCoords.y = Math.floor(Math.random() * this.gridY);
    }

    drawApple() {
        this.ctx.fillStyle = 'red';
        this.ctx.fillRect(this.appleCoords.x * this.sectionDimension, this.appleCoords.y * this.sectionDimension, this.sectionDimension - this.sectionOffset, this.sectionDimension - this.sectionOffset)
    }

    checkApple() {
        if (this.trail.length && this.trail[this.trail.length - 1].x == this.appleCoords.x && this.trail[this.trail.length - 1].y == this.appleCoords.y) {
            this.tail++;
            this.placeApple();
            this.appleReward = 10;
        }
    }

    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    die() {
        this.tail = 5;
        this.deadReward = -10;

    }

    reset() {
        this.gridX = this.canvas.width / this.sectionDimension - 1;
        this.gridY = this.canvas.height / this.sectionDimension - 1;
        // adjust the number of squares for the offset
        this.dir = {x: 1, y: 0};
        this.pos = {x: Math.floor(this.gridX / 2), y: Math.floor(this.gridY / 2)};
        this.trail = [];
    }

    saveBrain() {
        this.learnIteration++;
        if (this.learnIteration % 1000 == 0) { //save every 100 iterations
            var j = {
                learnIteration: this.learnIteration,
                network: this.brain.value_net.toJSON()
            };
            var t = JSON.stringify(j);
            window.localStorage.setItem('brain', t);
        }
    }

    loadBrain() {
        let save = window.localStorage.getItem('brain');
        if (save) {
            save = JSON.parse(save);
            console.log('Load save', save);
            this.learnIteration = save.learnIteration;
            this.network = save.network;
        }
    }
}

new

SnakeGame();