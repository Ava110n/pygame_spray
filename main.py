import random

import pygame
from sklearn.cluster import DBSCAN

class Point:
    def __init__(self, x, y, color="red"):
        self.x = x
        self.y = y
        self.color = color

colors = ["yellow", "blue", "green", "cyan", "pink", "brown", "orange", "red"]

def generate_points(center: tuple[int, int], eps: int = 30, min_n: int = 3, \
                    max_n: int = 7) -> list[tuple[int, int]]:
    return list(
        (center[0] + random.randint(-eps, eps),
         center[1] + random.randint(-eps, eps))
        for _ in range(random.randint(min_n, max_n))
    )


def main():
    pygame.init()

    screen = pygame.display.set_mode((640, 480), pygame.RESIZABLE)
    screen.fill("white")
    pygame.display.update()
    points = []
    while True:
        down = False
        mouse_pos = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                print(event.type)
                if event.button == 1:
                    down = True
                    mouse_pos = event.pos
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    down = False
            if event.type == pygame.WINDOWRESIZED:
                screen.fill("white")
                for x in points:
                    pygame.draw.circle(screen, "black", x, 5)
                #     pygame.draw.circle(screen, "red", event.pos, 5)
                #     points.append(event.pos)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    dbscan = DBSCAN(50, min_samples=3)
                    res = dbscan.fit_predict(points)
                    for i, x in enumerate(res):
                        pygame.draw.circle(screen, colors[x], points[i], 5)

            if down:
                if len(points) == 0 or (mouse_pos[0]-points[-1][0])**2 + \
                        (mouse_pos[1]-points[-1][1])**2 >= 400:
                    for x in generate_points(mouse_pos):
                        points.append(x)
                        pygame.draw.circle(screen, "black", x, 5)
                    points.append(mouse_pos)

        pygame.display.update()



if __name__ == '__main__':
    main()
