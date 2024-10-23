from PIL import Image, ImageDraw, ImageFilter
from random import randint
import time


#Should be map image
map_image = Image.open('map_image.png')
#Should be any drone placeholder image
drone_image = Image.open('test3.png')

resized_drone_img = drone_image.resize((50, 50)) 
modifiable_map_image = map_image

#Top left of map coordinates
top_left_longitude = -76.5566927
top_left_latitude = 38.319353

#Test coordinates (should be switched to actual inputs)
longitude = -76.548
latitude = 38.317

#Calculate difference for map position calculation
longitude_difference = top_left_longitude - longitude
latitude_difference = top_left_latitude - latitude

#Calculate pixel position on map image
longitude_map_position = (longitude_difference * -66293.37964) - 25
latitude_map_position = (latitude_difference * 84637.71678) - 25
longitude_map_position = int(longitude_map_position)
latitude_map_position = int(latitude_map_position)
#Confirm that expected coordinates were found
print(longitude_map_position, latitude_map_position)

#ADD: paste drone image on map at determined coordinates
modifiable_map_image.paste(resized_drone_img, (longitude_map_position, latitude_map_position))

modifiable_map_image.save('modded_map.png')

while(True):
  #Save updated map image with drone position 
  time.sleep(2)
  modifiable_map_image.save('modded_map.png')
  longitude_map_position = (randint(1, 500))
  latitude_map_position = (randint(1, 500))
  print(longitude_map_position, latitude_map_position)
  modifiable_map_image.paste(resized_drone_img, (longitude_map_position, latitude_map_position))


