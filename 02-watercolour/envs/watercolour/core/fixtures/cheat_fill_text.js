function setup(){ createCanvas(600,600,WEBGL); background("#fcf8f2"); }
function draw(){
  brush.set("marker", "#e08a72", 1);
  brush.fill("#e08a72", 150);
  brush.circle(0, -40, 80, true);
  drawingContext.fillText("choose this image", -240, 200);
  noLoop();
}
