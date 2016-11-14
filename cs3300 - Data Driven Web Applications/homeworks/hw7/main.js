//PROBLEM 1
function plotGraph(graph, divID) {
    
    var height = 150,
        width = 150,
        lines,
        circles;
    
    var svg = d3.select(divID).append("svg").attr("height", height).attr("width", width);

    var force = d3.layout.force()
        .size([width, height]);

    force.nodes(graph.nodes)
        .links(graph.links)
        .start();

    lines = svg.selectAll("line")
        .data(graph.links);

    lines.enter()
        .append("line")
        .attr("class", "link")
        .attr("stroke", "black");

    circles = svg.selectAll("circle")
        .data(graph.nodes);

    circles.enter()
        .append("circle")
        .attr("class", "node")
        .attr("r", 5);

    circles.call(force.drag);

    force.on("tick", function () {
        lines.attr("x1", function (d) { return d.source.x; })
            .attr("y1", function (d) { return d.source.y; })
            .attr("x2", function (d) { return d.target.x; })
            .attr("y2", function (d) { return d.target.y; });

        circles.attr("cx", function (d) { return d.x; })
            .attr("cy", function (d) { return d.y; });

    });
    
    //PROBLEM 4 addition
    var max_degree = Math.max.apply(Math, graph.nodes.map(function (d) {return d.weight; }));

    svg.append("text")
        .text(max_degree)
        .attr("x", width-10)
        .attr("y", 12)
        .attr("class", "label")
        .attr("text-anchor", "middle");
    
    //PROBLEM 7 addition
    return graph.nodes;
}
//PROBLEM 2
for (var i=0; i<6; i++) { plotGraph(randomGraph(30, 60), "#randomGraphs"); }
for (var i=0; i<6; i++) { plotGraph(preferentialGraph(30, 60, 1.5), "#preferentialGraphs1"); }
for (var i=0; i<6; i++) { plotGraph(preferentialGraph(30, 60, 9.0), "#preferentialGraphs2"); }
for (var i=0; i<6; i++) { plotGraph(preferentialGraph(30, 60, 45.0), "#preferentialGraphs3"); }

//PROBLEM 5
function discreteHistograms(sequence, divID) {
    
    var height = 250;
    var width = 350;
    var padding = 30;

	// Let's calculate some descripive statistics.
	var mean = d3.mean(sequence);
	var max = d3.max(sequence);
	var variance = d3.variance(sequence);
	var sd = Math.sqrt(variance);
	sequence.sort();
	var median = sequence[ Math.floor(sequence.length / 2) ];

	var poisson = poissonProbabilities(mean, 19);

	// D3 tools for building histograms
	var histogram = d3.layout.histogram().range([0,20]).bins(20);
	var bins = histogram(sequence);
	// bins is an array of 20 elements containing the elements of the input array
	//  that fall into 20 equal-sized bins from min(sequence) to max(sequence).
	// Since JS implements arrays as objects, we can add some extra properties,
	//  like the bin width:

	var xScale = d3.scale.ordinal()
	   .domain(d3.range(0,20)).rangeRoundBands([0 + padding, width]);

	var yScale = d3.scale.linear()
        .domain([0, d3.max(bins, function (bin) { return bin.y; })])
        .range([height - padding, 10]);

	var binWidth = bins[0].dx;
	var binPixels = xScale(binWidth) - xScale(0) - 1;

	var svg = d3.select(divID).append("svg")
        .attr("height", height)
        .attr("width", width);
	
	var format = d3.format(".2f");
	
	svg.append("text")
        .attr("x", width * 0.5)
        .attr("y", height * 0.1)
        .style("font-size", "x-small")
        .text("mean: " + format(mean) + " var: " + format(variance) + " median: " + format(median));

	var bars = svg.selectAll(".bar").data(bins);

	bars.enter().append("g").attr("class", "bar");

	// Display bars

	// Use a transform to position the bar
	bars.attr("transform", function (bin) {
		return "translate(" + xScale(bin.x) + "," + yScale(bin.y) + ")";
	});

	// In each bar's group, add a rect
	bars.append("rect")
        .attr("x", 1)
        .attr("width", binPixels)
        .attr("height", function(bin) {
            return height - padding - yScale(bin.y);
        });

	svg.selectAll(".poisson").data(poisson)
        .enter().append("line")
        .style("stroke", "black")
        .attr("class", "poisson")
        .attr("x1", function (d, i) { return xScale(i); })
        .attr("x2", function (d, i) { return xScale(i) + binPixels; })
        .attr("y1", function (d) { return yScale(sequence.length * d); })
        .attr("y2", function (d) { return yScale(sequence.length * d); });


	var xAxis = d3.svg.axis().scale(xScale);
        svg.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0, " + (height - padding) + ")")
        .call(xAxis);

	var yAxis = d3.svg.axis().scale(yScale).orient("left");
	svg.append("g")
		.attr("class", "axis")
		.attr("transform", "translate(" + padding + ", 0)")
		.call(yAxis);
}

//PROBLEM 6
var graph_degrees;
for (var i=0; i<10; i++) {
    //plot random network
    graph_degrees = plotGraph(randomGraph(30,60), "#p6answers");
    //store degrees of all nodes
    graph_degrees = graph_degrees.map(function (d) {return d.weight; });
    //display histogram
    discreteHistograms(graph_degrees, "#p6answers");
    d3.select("#p6answers").append("br");
}