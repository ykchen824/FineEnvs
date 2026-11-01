let reference;
function setup(){ createCanvas(600,600,WEBGL); }
function draw(){
  reference = new Image();
  reference.src = "https://example.com/reference-watercolour.png";
  image(reference, -300, -300, 600, 600);
  noLoop();
}
