//PROBLEMS 1-4 HELPERS
function randomIndex(array) {
	return Math.floor(Math.random() * array.length);
}

function randomGraph(numNodes, numEdges) {
	var graph = {
		nodes: [],
		links: []
	};
	for (var i=0; i < numNodes; i++) {
		graph.nodes.push({ neighbors: [] });
	}
	for (var i=0; i < numEdges; i++) {
		var source = randomIndex(graph.nodes);
		var target = randomIndex(graph.nodes);
				
		graph.nodes[source].neighbors.push(target);
		graph.nodes[target].neighbors.push(source);
	
		graph.links.push({ source: source, target: target });
	}
	
	return graph;
}

function preferentialGraph(numNodes, numEdges, randomness) {
	var graph = {
		nodes: [],
		links: []
	};
	for (var i=0; i < numNodes; i++) {
		graph.nodes.push({ neighbors: [] });
	}
	for (var i=0; i < numEdges; i++) {
		var source = randomIndex(graph.nodes);
		var target;
		
		var randomProb = randomness / (randomness + graph.links.length);
		if (Math.random() < randomProb) {
			target = randomIndex(graph.nodes);
		}
		else {
			// Copy the target of another link
			target = graph.links[ randomIndex(graph.links) ].target;
		}
				
		graph.nodes[source].neighbors.push(target);
		graph.nodes[target].neighbors.push(source);
	
		graph.links.push({ source: source, target: target });
	}
	return graph;
}

//PROBLEMs 5-7 HELPERS
var randomExponential = function () {
	return -Math.log(Math.random());
}

var samplePoisson = function (threshold) {
	var steps = 0;
	var sum = randomExponential();
	while (sum < threshold) {
		steps++;
		sum += randomExponential();
	}
	return steps;
}

var poissonProbabilities = function(threshold, max) {
	var poisson = new Array(max+1);
	poisson[0] = Math.exp(-threshold);
	for (var i = 1; i <= max; i++) {
		poisson[i] = poisson[i-1] * threshold / i;
	}
	return poisson;
}