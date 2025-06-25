from vision.mapping_pipeline import mapping_pipeline
import asyncio

if __name__ == "__main__":
    print("AHH")
    asyncio.run(mapping_pipeline("./flight/data/mapping_photos.json", "mapping_images", "", ""))
