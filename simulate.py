import pygame

from pendulum import cart_and_bob_points
from solve import build_closed_case, build_control_case

MODE = "closed"

W, H = 1000, 600
FPS = 60
SCALE = 120
GX, GY, GW, GH = 20, 20, 960, 140
CART_TOP = 380
CART_W = 90
CART_H = 34
WHEEL_R = 9
GROUND_Y = CART_TOP + CART_H + 14


def load_case():
    if MODE == "closed":
        return build_closed_case()
    if MODE == "control":
        return build_control_case()
    raise ValueError("MODE must be 'closed' or 'control'")


def graph_data(values):
    low, high = values.min(), values.max()
    if high - low < 1e-9:
        high = low + 1.0
    points = [
        (GX + int(GW * index / (len(values) - 1)), GY + GH - int(GH * (value - low) / (high - low)))
        for index, value in enumerate(values)
    ]
    zero = None if not low <= 0.0 <= high else GY + GH - int(GH * (-low) / (high - low))
    return points, zero


def draw(screen, case, graph_points_data, zero_line, frame):
    trajectory = case["trajectory"]
    state = trajectory.solution[frame, 1:]
    cart_x, bob_x, bob_y = cart_and_bob_points(state, case["params"], case["equilibrium_sign"])
    cx = W // 2 + int(cart_x * SCALE)
    bx = W // 2 + int(bob_x * SCALE)
    by = CART_TOP + int(bob_y * SCALE)

    screen.fill((245, 245, 245))
    pygame.draw.rect(screen, (255, 255, 255), (GX, GY, GW, GH))
    pygame.draw.rect(screen, (180, 180, 180), (GX, GY, GW, GH), 1)
    if zero_line is not None:
        pygame.draw.line(screen, (220, 220, 220), (GX, zero_line), (GX + GW, zero_line), 1)
    pygame.draw.lines(screen, (60, 100, 170), False, graph_points_data, 2)
    pygame.draw.circle(screen, (200, 70, 70), graph_points_data[frame], 4)

    pygame.draw.line(screen, (150, 150, 150), (0, GROUND_Y), (W, GROUND_Y), 2)
    pygame.draw.rect(screen, (60, 100, 170), (cx - CART_W // 2, CART_TOP, CART_W, CART_H), border_radius=8)
    pygame.draw.circle(screen, (60, 60, 60), (cx - 24, GROUND_Y - WHEEL_R), WHEEL_R)
    pygame.draw.circle(screen, (60, 60, 60), (cx + 24, GROUND_Y - WHEEL_R), WHEEL_R)
    pygame.draw.circle(screen, (35, 35, 35), (cx, CART_TOP), 5)
    pygame.draw.line(screen, (30, 30, 30), (cx, CART_TOP), (bx, by), 4)
    pygame.draw.circle(screen, (200, 70, 70), (bx, by), 13)


def main():
    case = load_case()
    graph_points_data, zero_line = graph_data(case["graph_values"])
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(case["name"])
    clock = pygame.time.Clock()
    frame = 0
    frame_count = len(case["trajectory"].t)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        draw(screen, case, graph_points_data, zero_line, frame)
        pygame.display.flip()
        frame = (frame + 1) % frame_count
        clock.tick(FPS)


if __name__ == "__main__":
    main()
