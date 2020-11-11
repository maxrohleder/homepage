function setup() {
	var scale = min(displayWidth, 800);
	var canvas = createCanvas(scale, scale/2);
	canvas.parent('XorSketch-div');
  	
  	// initialise weights from in to hidden
	for(var i = 0; i < 3; i++){
		for (var j = 0; j < 3; j++) {
			w1[i][j] = -1+random(2);
		}
	}

	// init from hidden to out
	for(i = 0; i < 4; i++){
		w2[i] = -1+random(2);
	}
}

// init outputs from each neuron --> modified by feedforward method
var inputLayer = [0, 1];
var hiddenLayer = [1.0, 1.0, 1.0];
var outputLayer = [1];

// weights from in to hid and hid to out
// to be optimized by train method
var w1 =  [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
var w2 = [0, 0, 0, 0];

// trainings to do --> modified by buttons
var trainNum = 0;

// connections to learn: [in1, in2, out1] whole network is a function from R^2 to R
var trainingSets = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];

//setting the learningrate to 0.5 (experience shows its a good value)
var lr = 0.5;

//init documentation to be drawn later
var docu = '';

//amount of performed trainings
var trainCounter = 0;

//main method
function draw() {
	// all vizualisation extracted in method
	drawing();

	//train network on examples
	if(trainNum !== 0){
		var trainNum1 = trainNum;
		while(trainNum != 0 && (trainNum1-trainNum) < 20){
			train(Math.round(random(3)));
			trainNum--;
		}
		// performance is logged
		logtotalError();
		testwithexamples();
	}
}

function drawing(){
		
		background(50);
		var input_x = width/6;
		var hidden_x = width/2;
		var out_x = width*5/6;

		//draw w1
		for (var i = w1[0].length - 1; i >= 0; i--) {
			noStroke();
			fill(255);
			textSize(width/70);
			textAlign(CENTER, CENTER);
			text(nfc(w1[0][i],3), width/2, (height*(i+1)/4)-(width/16)*0.7); 
		};
		for(var i = 1; i < 3; i++){
			for(var j = 0; j < 3; j++){
				strokeWeight(Math.abs(w1[i][j]) + 1);
				// coloring blue or red depending on sign of weight
				if(w1[i][j] < 0){
					stroke(255,0,0);
				}else{
					stroke(0, 0, 255);	
				}
				line(input_x, height*i/3, hidden_x, height*(j+1)/4);
				
				noStroke();
				fill(255);
				textAlign(LEFT, CENTER);
				text(nfc(w1[i][j], 3), (input_x+hidden_x)*6/10, ((height*i/3)+3*(height*(j+1)/4))/4);
			}
		}
		
		//draw w2
		for(i = 3; i > 0; i--){
			strokeWeight(Math.abs(w2[i]) + 1);
			if(w2[i] < 0){
					stroke(255,0,0);
			}else{
					stroke(0, 0, 255);	
			}
			line(hidden_x, height*i/4, out_x, height/2);

			noStroke();
			fill(255);
			textStyle(NORMAL);
			textSize(width/70);
			textAlign(CENTER, CENTER);
			// draw weight on lines
			text(nfc(w2[i], 3), (out_x+hidden_x)/2, height*((i*0.125)+0.25));
		}
		textAlign(CENTER, CENTER);
		textSize(width/70);

		// draw out bias
		text(nfc(w2[0], 3), out_x, (height/2)-(width/16)*0.7);

		//draw Input Nodes
		for(i = 0; i < inputLayer.length; i++){
			fill(255, 160, 40);
			ellipse(input_x, height*(i+1)/(inputLayer.length+1), width/16, width/16);
		}
		
		//draw Hidden Nodes
		for(i = 0; i < hiddenLayer.length; i++){
			fill(255, 160, 40);
			ellipse(hidden_x, height*(i+1)/(hiddenLayer.length+1), width/16, width/16);
		}		

		//draw Output Node
		fill(255, 160, 40);
		ellipse(out_x, height/2, width/16, width/16);

		//draw documentation
		noStroke();
		fill(255);
		textSize(width/70);
		textAlign(LEFT, CENTER);
		text(docu, width*5/6, height*4/5);

		//draw trainCounter
		noStroke();
		fill(255);
		textAlign(CENTER, CENTER);
		text('times trained:  ' + trainCounter, input_x, height*0.9);	

		//draw Error
		fill(255, 160, 40);
		noStroke();
		textSize(width/30);
		textAlign(CENTER, CENTER);
		text('ERROR: ' + nfc(logtotalError(), 5), width/2, height*9/10);	
}

function train1time(){
	trainNum += 1;
}

function train1000times(){
	trainNum += 10000;
}

function reset(){
		trainCounter = 0;
		trainNum = 0;
		// initialise weights from in to hidden
		for(var i = 0; i < 3; i++){
			for (var j = 0; j < 3; j++) {
				w1[i][j] = -1 + random(2);
			}
  		}
  		// init from hidden to out
 		for(i = 0; i < 4; i++){
  			w2[i] = -1 + random(2);
		}
		logtotalError();
		testwithexamples();  
}

function testwithexamples(){
	docu = 'SYSTEM ANSWERS:\n';
	for (var i = trainingSets.length - 1; i >= 0; i--) {
		docu += trainingSets[i][0];
		docu += '  xor  '
		docu += trainingSets[i][1];
		feedforward(i);
		docu += '  =  ' + nfc(outputLayer[0], 3) + '\n';
	};
}

function train(a){
	feedforward(a);
	trainCounter++;

	// mean squares derivate
	var E = outputLayer[0] - trainingSets[a][2]; 
	
	// logistics derivate
	var Ek0 = (outputLayer[0] * (1 - outputLayer[0])) * E;

	//console.log('Ek0: ' + Ek0);

	// backpropagated error of all hidden neurons with chain rule
	var j0_incoming = w2[1] * Ek0;
	var j1_incoming = w2[2] * Ek0;
	var j2_incoming = w2[3] * Ek0;

	// translate error through nonlinear logistic function
	var Ej0 = (hiddenLayer[0] * (1 - hiddenLayer[0])) * j0_incoming;
	var Ej1 = (hiddenLayer[1] * (1 - hiddenLayer[1])) * j1_incoming;
	var Ej2 = (hiddenLayer[2] * (1 - hiddenLayer[2])) * j2_incoming;

	//console.log('Ej0:   ' + Ej0);
	//console.log('Ej1:   ' + Ej1);
	//console.log('Ej2:   ' + Ej2);

	//update all weights
	w2[0] -= lr * Ek0;
	w2[1] -= lr * Ek0 * hiddenLayer[0];
	w2[2] -= lr * Ek0 * hiddenLayer[1];
	w2[3] -= lr * Ek0 * hiddenLayer[2];

	w1[0][0] -= lr * Ej0;
	w1[1][0] -= lr * Ej0 * inputLayer[0];
	w1[2][0] -= lr * Ej0 * inputLayer[1];

	w1[0][1] -= lr * Ej1;
	w1[1][1] -= lr * Ej1 * inputLayer[0];
	w1[2][1] -= lr * Ej1 * inputLayer[1];

	w1[0][2] -= lr * Ej2;
	w1[1][2] -= lr * Ej2 * inputLayer[0];
	w1[2][2] -= lr * Ej2 * inputLayer[1];
}

function feedforward(a){
	//set inputlayer from trainingsets
	for (var i = 1; i >= 0; i--) {
		inputLayer[i] = trainingSets[a][i];
	}

	//compute hiddenlayer net input wij (i = 0 --> bias)
	var j0net = w1[0][0] + (inputLayer[0]*w1[1][0]) + (inputLayer[1]*w1[2][0]);
	var j1net = w1[0][1] + (inputLayer[0]*w1[1][1]) + (inputLayer[1]*w1[2][1]);
	var j2net = w1[0][2] + (inputLayer[0]*w1[1][2]) + (inputLayer[1]*w1[2][2]);

	//compute hiddenlayer out with logistic function
	hiddenLayer[0] = (1/(1+exp(-j0net)));
	hiddenLayer[1] = (1/(1+exp(-j1net)));
	hiddenLayer[2] = (1/(1+exp(-j2net)));

	//compute output net Input wjk
	var k0net = w2[0] + (hiddenLayer[0]*w2[1]) + (hiddenLayer[1]*w2[2]) + (hiddenLayer[2]*w2[3]);

	//compute output logistic
	outputLayer[0] = 1/(1+exp(-k0net));
}

function logtotalError(){
	var totalError = 0;
	for (var i = 3; i >= 0; i--) {
		feedforward(i);
		totalError += (Math.pow((trainingSets[i][2] - outputLayer[0]), 2)*(1/2));
	}
	//console.log('Overall Performance:  ' + totalError);	
	return totalError;
}