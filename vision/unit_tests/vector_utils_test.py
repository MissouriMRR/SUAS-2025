from vision.deskew import vector_utils

"""pixel_intersect(
    pixel: tuple[int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
    rotation_deg: list[float],
    height: float,
) -> Point | None:"""

print(vector_utils.pixel_intersect((1920, 1080), (1080, 1920), [0, -90, 0], 100))
