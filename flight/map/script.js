function loadImage(path){
  let image = new Image();
  image.src = path;
  setTimeout(function(){}, 100)
  return image;
}

setInterval(function() {
 // var myImageElement = document.getElementById('map_image');
  myImageElement = loadImage('modded_map.png?rand=' + Math.random());
}, 3000);

//AHHH FLICKER AHHH
// change html to not multiply cats