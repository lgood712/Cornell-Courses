//	Luke Goodman (lng 26) - INFO3300 HW3

// Problem 1
d3.selectAll(".findable")
	.text("found")
	.style("color", "red");

// Problem 2
var height = 300;
var width = 300;
var padding = 50;

var circles = [{'x':2,'y':4}, {'x':1,'y':1}], 
	squares = [{'x':4,'y':15}, {'x':9,'y':80}];
//var dataset = [{'x':25,'y':11}, {'x':0,'y':0}, {'x':75,'y':53}, {'x':200,'y':300}];

var minMax = {'xmin': 1, 'xmax': 9, 'ymin': 1, 'ymax': 80};

var svg = d3.select("#wick").append("svg")
   .attr("height", height).attr("width", width);

var xScale = d3.scale.linear()
  .domain([minMax.xmin-1, minMax.xmax+1]).range([0, 200]);

var yScale = d3.scale.linear()
  .domain([minMax.ymin-10, minMax.ymax + 10]).range([200, 0]);

var vert_pos = yScale(minMax.ymin) + padding + 20;
var xAxis = d3.svg.axis().scale(xScale);
svg.append("g")
  .attr("class", "axis")
  .attr("transform", "translate("+padding+", " + vert_pos + ")")
  .call(xAxis);

var yAxis = d3.svg.axis().scale(yScale).orient("left");
svg.append("g")
	.attr("class", "axis")
	.attr("transform", "translate(" + padding + ", "+padding+")")
	.call(yAxis);

svg.selectAll("circle")
	.data(circles).enter()
	.append("circle")
	.attr("cx", function(d) { return padding + xScale(d.x); })
	.attr("cy", function(d) { return padding + yScale(d.y); })
	.attr("r", 7)
	.on("click", function() { d3.select(this).style("fill", "blue"); });
svg.selectAll("rect")
	.data(squares).enter()
	.append("rect")
	.attr("x", function(d) { return padding + xScale(d.x); })
	.attr("y", function(d) { return padding + yScale(d.y); })
	.attr("height", 14)
	.attr("width", 14)
	.on("click", function() { d3.select(this).style("fill", "blue"); });
svg.append("text")
	.attr("x", 130)
	.attr("y", 35)
	.text("Title");
svg.append("text")
	.attr("x", 5)
	.attr("y", height/2)
	.text("C");
svg.append("text")
	.attr("x", width/2)
	.attr("y", height-10)
	.text("A");
	
// Problem 3
// from zipf.json
var wordRanks = [{rank: 1, count: 15342397280, word: "of"},
{rank: 2, count: 7611765281, word: "in"},
{rank: 4, count: 11021132912, word: "and"},
{rank: 8, count: 3021925527, word: "for"},
{rank: 16, count: 1562321315, word: "at"},
{rank: 32, count: 746702714, word: "more"},
{rank: 64, count: 652703221, word: "its"},
{rank: 128, count: 186584246, word: "different"},
{rank: 256, count: 102237666, word: "light"},
{rank: 512, count: 44824164, word: "middle"},
{rank: 1024, count: 34340679, word: "additional"},
{rank: 2048, count: 11864923, word: "abandoned"},
{rank: 4096, count: 7506730, word: "motive"},
{rank: 8192, count: 10615754, word: "Die"},
{rank: 16384, count: 907890, word: "portrays"},
{rank: 32768, count: 226851, word: "Fundamentally"},
{rank: 65536, count: 144913, word: "dv"}];

// 3A
var svgA = d3.select("#threeA").append("svg")
			.attr("height", 200)
			.attr("width", 200);

var xScale = d3.scale.linear()
  .domain([0,66000]).range([0, 200]);

var yScale = d3.scale.linear()
  .domain([100000,16000000000]).range([200,0]);

var wordGroup = svgA.append("g"),
	words = wordGroup.selectAll("text").data(wordRanks);

words.enter().append("text")
	.attr("x", function(w) { return xScale(w.rank); })
	.attr("y", function(w) { return yScale(w.count); })
	.text(function (w) { return w.word; } );

// 3B
var svgB = d3.select("#threeB").append("svg")
			.attr("height", 200)
			.attr("width", 200);

var xScaleB = d3.scale.linear()
  .domain([0,Math.log(66000)]).range([0, 200]);

var yScaleB = d3.scale.linear()
  .domain([Math.log(100000),Math.log(16000000000)]).range([200,0]);

var wordGroupB = svgB.append("g"),
	wordsB = wordGroupB.selectAll("text").data(wordRanks);

wordsB.enter().append("text")
	.attr("x", function(w) { return xScaleB(Math.log(w.rank)); })
	.attr("y", function(w) { return yScaleB(Math.log(w.count)); })
	.text(function (w) { return w.word; } );

// 3C
var svgC = d3.select("#threeC").append("svg")
			.attr("height", 200)
			.attr("width", 200);

var xScaleC = d3.scale.log()
  .domain([1,66000]).range([0, 200]);

var yScaleC = d3.scale.log()
  .domain([100000,16000000000]).range([200,0]);

var wordGroupC = svgC.append("g"),
	wordsC = wordGroupC.selectAll("text").data(wordRanks);

wordsC.enter().append("text")
	.attr("x", function(w) { return xScaleC(w.rank); })
	.attr("y", function(w) { return yScaleC(w.count); })
	.text(function (w) { return w.word; } );

	
// Problem 4
var width4 = 960,
    height4 = 500;

var projection = d3.geo.albersUsa();

var path = d3.geo.path().projection(projection);
	
var svg4 = d3.select("#four").append("svg")
    .attr("width", width4)
    .attr("height", height4);

var states;
d3.json("us.json", function(error, shapes) {
	states = topojson.feature(shapes, shapes.objects.states).features;
	var statePaths = svg4.append("g");
	statePaths.selectAll("path").data(states).enter()
		.append("path").attr("d", path)
		.style("fill", "none").style("stroke", "#ccc");
	
	var locations = [{"city":"New York City", "coords":projection([-74,41])},
					{"city":"Raleigh", "coords":projection([-78.6,35.8])},
					{"city":"Dallas", "coords":projection([-96.8,32.8])}];
	
	svg4.selectAll("circle")
		.data(locations).enter()
		.append("circle")
		.attr("cx", function(d) { return d.coords[0]; } )
		.attr("cy", function(d) { return d.coords[1]; })
		.attr("r", 5)
		.style("fill", "red");

	svg4.selectAll("text")
		.data(locations).enter()
		.append("text")
		.attr("x", function(d) { return d.coords[0]; } )
		.attr("y", function(d) { return d.coords[1]; } )
		.text( function(d) { return d.city; } );
});

