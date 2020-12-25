var map = document.getElementById('myCanvas');
paper.setup(map);

function emoji(radius, position, vector) {
	this.radius = radius;
	this.point = position;
	this.vector = vector;
	this.maxVec = 15;
	this.numSegment = Math.floor(this.radius / 3 + 2);
	this.boundOffset = [];
	this.boundOffsetBuff = [];
	this.sidePoints = [];
	this.path = new paper.Path({
		fillColor: {
			hue: 60,
			saturation: 1,
			brightness: 1
		},
		blendMode: 'lighter'
	});

	for (var i = 0; i < this.numSegment; i ++) {
		this.boundOffset.push(this.radius);
		this.boundOffsetBuff.push(this.radius);
		this.path.add(new paper.Point());
		this.sidePoints.push(new paper.Point({
			angle: 360 / this.numSegment * i,
			length: 1
		}));
	}
}

emoji.prototype = {
	iterate: function() {
		this.checkBorders();
		if (this.vector.length > this.maxVec)
			this.vector.length = this.maxVec;
		this.point += this.vector;
		this.updateShape();
	},

	checkBorders: function() {
		var size = view.size;
		if (this.point.x < -this.radius)
			this.point.x = size.width + this.radius;
		if (this.point.x > size.width + this.radius)
			this.point.x = -this.radius;
		if (this.point.y < -this.radius)
			this.point.y = size.height + this.radius;
		if (this.point.y > size.height + this.radius)
			this.point.y = -this.radius;
	},

	updateShape: function() {
		var segments = this.path.segments;
		for (var i = 0; i < this.numSegment; i ++)
			segments[i].point = this.getSidePoint(i);

		this.path.smooth();
		for (var i = 0; i < this.numSegment; i ++) {
			if (this.boundOffset[i] < this.radius / 4)
				this.boundOffset[i] = this.radius / 4;
			var next = (i + 1) % this.numSegment;
			var prev = (i > 0) ? i - 1 : this.numSegment - 1;
			var offset = this.boundOffset[i];
			offset += (this.radius - offset) / 15;
			offset += ((this.boundOffset[next] + this.boundOffset[prev]) / 2 - offset) / 3;
			this.boundOffsetBuff[i] = this.boundOffset[i] = offset;
		}
	},

	react: function(b) {
		var dist = this.point.getDistance(b.point);
		if (dist < this.radius + b.radius && dist != 0) {
			var overlap = this.radius + b.radius - dist;
			var direc = (this.point - b.point).normalize(overlap * 0.015);
			this.vector += direc;
			b.vector -= direc;

			this.calcBounds(b);
			b.calcBounds(this);
			this.updateBounds();
			b.updateBounds();
		}
	},

	getBoundOffset: function(b) {
		var diff = this.point - b;
		var angle = (diff.angle + 180) % 360;
		return this.boundOffset[Math.floor(angle / 360 * this.boundOffset.length)];
	},

	calcBounds: function(b) {
		for (var i = 0; i < this.numSegment; i ++) {
			var tp = this.getSidePoint(i);
			var bLen = b.getBoundOffset(tp);
			var td = tp.getDistance(b.point);
			if (td < bLen) {
				this.boundOffsetBuff[i] -= (bLen  - td) / 2;
			}
		}
	},

	getSidePoint: function(index) {
		return this.point + this.sidePoints[index] * this.boundOffset[index];
	},

	updateBounds: function() {
		for (var i = 0; i < this.numSegment; i ++)
			this.boundOffset[i] = this.boundOffsetBuff[i];
	}
};

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

var emojis = [];
var numEmojis = 1;
for (var i = 0; i < numEmojis; i++) {
	var position = new paper.Point(width / 2, height / 2);
	var vector = new paper.Point({
		angle: 0.01,
		length: 0
	});
	var radius = 100;
	emojis.push(new emoji(radius, position, vector));
}

function onFrame() {
	for (var i = 0; i < emojis.length - 1; i++) {
		for (var j = i + 1; j < emojis.length; j++) {
			emojis[i].react(emojis[j]);
		}
	}
	for (var i = 0, l = emojis.length; i < l; i++) {
		emojis[i].iterate();
	}
}

// render
paper.view.draw();