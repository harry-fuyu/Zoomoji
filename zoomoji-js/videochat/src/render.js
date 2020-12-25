var map = document.getElementById('myCanvas');
paper.setup(map);

var width  = paper.view.size.width;
var height = paper.view.size.height;
var mouthLocs = [{ "x": width / 2 - 30, "y": height / 2 + 30 },
{ "x": width / 2, "y": height / 2 + 50 },
{ "x": width / 2 + 30, "y": height / 2 + 30 }];

var circle = new paper.Shape.Circle({
  center: [width/2, height/2],
  fillColor: 'yellow',
  radius: 100
});

var eyesLeft = new paper.Shape.Circle({
	center: [width / 2 - 20, height / 2 - 20],
	fillColor: 'black',
	radius: 5
});

var eyesRight = new paper.Shape.Circle({
	center: [width / 2 + 20, height / 2 - 20],
	fillColor: 'black',
	radius: 5
});

var mouthPoints = [];
for (i = 0; i < mouthLocs.length; ++i){
	mouthPoints.push(new paper.Point(mouthLocs[i]["x"], mouthLocs[i]["y"]));
}
var mouth = new paper.Path.Arc(mouthPoints[0], mouthPoints[1], mouthPoints[2]);
mouth.strokeWeight = 2;
mouth.strokeColor = 'black';

// render
paper.view.draw();