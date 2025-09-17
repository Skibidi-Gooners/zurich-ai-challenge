import os, random, copy, pygame, neat

# ================== Setup & Konstanten ==================
pygame.init()
SCREEN_W, SCREEN_H = 1100, 600
SCREEN = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("T-Rex Runner NEAT")
FONT = pygame.font.Font('freesansbold.ttf', 20)

# Gameplay
BASE_SPEED, MAX_SPEED, SPEED_STEP_POINTS, FPS = 20, 60, 100, 30
REACTION_FRAMES, FLIGHT_FRAMES = 10, 22
GAP_MULT, INTRA_MULT = 2.8, 2.2

# Gene/Netz
TRAITS_DIM = 3
DEFAULT_INPUTS = 15 + TRAITS_DIM  # y,d1,h1,w1,speed,d2,h2,w2,t1,t2,gap,s1,l1,s2,l2 + traits(3) = 18

# ================== Prozedurale Assets ==================
def generate_assets():
    """
    Erzeugt alle Sprite-Surfaces (RUN[2], JUMP, SC[2], LC[2], BG) rein per Pygame-Primitives.
    Größen sind auf die Bodenlinie bei y=380 abgestimmt:
      - Dino:   88x70, steht bei Y=310 (310+70=380)
      - Small:  ~55px hoch, Y=325
      - Large:  ~80px hoch, Y=300
      - Track:  1200x16, wird bei y=380 geblittet und horizontal getiled
    """
    DINO = (50, 50, 50)
    EYE  = (15, 15, 15)
    CACT = (36, 92, 52)
    GROUND = (110, 110, 110)
    DUST = (160, 160, 160)

    def dino_frame(leg_phase=0):
        w, h = 88, 70
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        # Rumpf / Kopf / Schnauze / Schwanz
        pygame.draw.rect(s, DINO, pygame.Rect(10, 28, 42, 22))
        pygame.draw.rect(s, DINO, pygame.Rect(42, 14, 20, 16))
        pygame.draw.rect(s, DINO, pygame.Rect(58, 20, 10, 8))
        pygame.draw.polygon(s, DINO, [(10, 40), (0, 36), (0, 44)])
        # Auge + Arm
        pygame.draw.rect(s, EYE,  pygame.Rect(52, 18, 3, 3))
        pygame.draw.rect(s, DINO, pygame.Rect(36, 32, 8, 3))
        # Beine (zwei Phasen)
        if leg_phase == 0:
            pygame.draw.rect(s, DINO, pygame.Rect(18, 50, 10, 18))
            pygame.draw.rect(s, DINO, pygame.Rect(38, 52, 10, 16))
        else:
            pygame.draw.rect(s, DINO, pygame.Rect(18, 52, 10, 16))
            pygame.draw.rect(s, DINO, pygame.Rect(38, 50, 10, 18))
        return s

    def dino_jump():
        s = pygame.Surface((88, 70), pygame.SRCALPHA)
        pygame.draw.rect(s, DINO, pygame.Rect(10, 26, 42, 22))
        pygame.draw.rect(s, DINO, pygame.Rect(42, 12, 20, 16))
        pygame.draw.rect(s, DINO, pygame.Rect(58, 18, 10, 8))
        pygame.draw.polygon(s, DINO, [(10, 38), (0, 34), (0, 42)])
        pygame.draw.rect(s, EYE,  pygame.Rect(52, 16, 3, 3))
        pygame.draw.rect(s, DINO, pygame.Rect(36, 30, 8, 3))  # Arm
        pygame.draw.rect(s, DINO, pygame.Rect(22, 48, 10, 14))  # Beine kompakt
        pygame.draw.rect(s, DINO, pygame.Rect(36, 48, 10, 14))
        return s

    RUN = [dino_frame(0), dino_frame(1)]
    JUMP = dino_jump()

    def cactus_surface(width, height, segments):
        s = pygame.Surface((width, height), pygame.SRCALPHA)
        for (x, y_from_bottom, w, h) in segments:
            top = height - y_from_bottom - h
            pygame.draw.rect(s, CACT, pygame.Rect(x, top, w, h), border_radius=2)
        # kleine Stachel-/Farbakzente
        for i in range(6):
            px = 4 + (i * (width - 8) // 5)
            py = height - 6 - (i % 3) * 6
            if 0 <= px < width and 0 <= py < height:
                s.set_at((px, py), (20, 60, 35, 255))
        return s

    # Kleine Kakteen (~55px)
    def small_cactus_v0():
        w, h = 36, 55
        seg = [(16, 0, 6, 48), (6, 22, 6, 16), (24, 18, 6, 20)]
        return cactus_surface(w, h, seg)

    def small_cactus_v1():
        w, h = 42, 55
        seg = [(18, 0, 6, 50), (8, 16, 6, 18), (28, 24, 6, 16)]
        return cactus_surface(w, h, seg)

    # Große Kakteen (~80px)
    def large_cactus_v0():
        w, h = 50, 80
        seg = [(22, 0, 6, 72), (10, 20, 6, 22), (34, 28, 6, 26), (4, 34, 6, 16), (40, 16, 6, 20)]
        return cactus_surface(w, h, seg)

    def large_cactus_v1():
        w, h = 60, 80
        seg = [(28, 0, 6, 74), (16, 14, 6, 26), (40, 22, 6, 24), (6, 30, 6, 18), (50, 18, 6, 18)]
        return cactus_surface(w, h, seg)

    SC = [small_cactus_v0(), small_cactus_v1()]
    LC = [large_cactus_v0(), large_cactus_v1()]

    # Track / Bodenlinie
    def make_track(width=1200, height=16):
        s = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.line(s, GROUND, (0, height - 4), (width, height - 4), 2)
        rng = random.Random(1337)
        for _ in range(140):
            x = rng.randint(0, width - 2)
            y = height - 6 + rng.randint(-3, 1)
            pygame.draw.rect(s, DUST, pygame.Rect(x, y, 2, 2))
        for x in range(0, width, 40):
            pygame.draw.line(s, GROUND, (x + 10, height - 8), (x + 22, height - 8), 1)
        return s

    BG = make_track(1200, 16)
    return RUN, JUMP, SC, LC, BG

RUN, JUMP, SC, LC, BG = generate_assets()

# ================== Entities ==================
class Dino:
    X, Y, J = 80, 310, 8.5  # Start X/Y, Jump-Startgeschwindigkeit
    def __init__(s):
        s.image, s.rect = RUN[0], RUN[0].get_rect()
        s.rect.x, s.rect.y = s.X, s.Y
        s.v, s.run, s.jump, s.step = s.J, True, False, 0

    def update(s):
        if s.jump:
            s.image = JUMP
            s.rect.y -= s.v * 4
            s.v -= 0.8
            if s.v < -s.J:
                s.jump = False
                s.v = s.J
                s.run = True
        else:
            s.image = RUN[s.step // 5]
            s.rect.x, s.rect.y = s.X, s.Y
            s.step = (s.step + 1) % 10

    def draw(s):
        SCREEN.blit(s.image, s.rect)

class Obs:
    def __init__(s, imgs, typ, kind, y):
        s.imgs, s.typ, s.kind = imgs, typ, kind
        s.rect = imgs[typ].get_rect()
        s.rect.x, s.rect.y = SCREEN_W, y
    def upd(s, spd):
        s.rect.x -= spd
        return s.rect.x < -s.rect.width
    def draw(s):
        SCREEN.blit(s.imgs[s.typ], s.rect)

class Small(Obs):
    def __init__(s): super().__init__(SC, random.randint(0, 1), "small", 325)

class Large(Obs):
    def __init__(s): super().__init__(LC, random.randint(0, 1), "large", 300)

# ================== Spawner ==================
class Spawner:
    def __init__(s):
        s.r = random.Random()
        s.next_x = SCREEN_W + 600
        s.hard = False

    @staticmethod
    def min_gap(spd):
        return max(300, int(spd * (REACTION_FRAMES + int(0.6 * FLIGHT_FRAMES)) * GAP_MULT))

    @staticmethod
    def seq_gap(spd):
        return max(240, int(spd * int(0.6 * FLIGHT_FRAMES) * INTRA_MULT))

    def _single(s):     return [Small()] if s.r.randint(0, 1) == 0 else [Large()]
    def _double(s, spd):
        a = s._single()
        b = s._single()[0]
        b.rect.x = SCREEN_W + s.seq_gap(spd)
        a.append(b)
        return a
    def _triple(s, spd):
        a = s._double(spd)
        c = s._single()[0]
        c.rect.x = SCREEN_W + 2 * s.seq_gap(spd)
        a.append(c)
        return a

    def _schedule(s, right, spd):
        g = s.min_gap(spd)
        s.next_x = max(SCREEN_W + 120, right + s.r.randint(int(g * 1.0), int(g * 1.7)))

    def maybe_spawn(s, arr, spd, epoch):
        last_right = (arr[-1].rect.x + arr[-1].rect.width) if arr else 0
        if last_right > s.next_x:
            return
        mod, roll = epoch % 3, s.r.random()
        if mod == 1:
            pack = s._single(); s.hard = False
        elif mod == 2:
            pack = s._single() if (roll < 0.75 or s.hard) else s._double(spd); s.hard = len(pack) > 1
        else:
            if s.hard:
                pack = s._single() if roll < 0.6 else s._double(spd); s.hard = len(pack) > 1
            else:
                pack = s._single() if roll < 0.55 else (s._double(spd) if roll < 0.9 else s._triple(spd))
                s.hard = len(pack) > 1

        base = max(SCREEN_W + 120,
                   (arr[-1].rect.x + arr[-1].rect.width + s.min_gap(spd)) if arr else SCREEN_W + s.min_gap(spd))
        pack[0].rect.x = base
        if len(pack) > 1: pack[1].rect.x = base + (pack[1].rect.x - SCREEN_W)
        if len(pack) > 2: pack[2].rect.x = base + (pack[2].rect.x - SCREEN_W)

        arr.extend(pack)
        s._schedule(pack[-1].rect.x + pack[-1].rect.width, spd)

# ================== Utils ==================
def scroll_bg(x, spd):
    w = BG.get_width()
    SCREEN.blit(BG, (x, 380))
    SCREEN.blit(BG, (w + x, 380))
    if x <= -w:
        x = 0
    return x - spd

def hud(points, epoch, spd):
    for i, t in enumerate((f"Points: {points}", f"Epoch: {epoch}", f"Speed: {spd}")):
        SCREEN.blit(FONT.render(t, True, (0, 0, 0)), (950, 30 + i * 30))

def build_inputs(net, dino, o1, o2, spd, traits):
    y = float(dino.rect.y)
    speed = float(spd)

    if o1:
        d1 = o1.rect.x - dino.rect.x; w1 = o1.rect.width; h1 = o1.rect.height
        s1 = 1.0 if o1.kind == "small" else 0.0; l1 = 1.0 - s1
        t1 = d1 / max(1e-3, spd)
    else:
        d1, w1, h1, s1, l1, t1 = float(SCREEN_W), 0, 0, 0.0, 0.0, 1e3

    if o2:
        d2 = o2.rect.x - dino.rect.x; w2 = o2.rect.width; h2 = o2.rect.height
        s2 = 1.0 if o2.kind == "small" else 0.0; l2 = 1.0 - s2
        t2 = d2 / max(1e-3, spd)
        gap = o2.rect.x - (o1.rect.x + o1.rect.width) if o1 else SCREEN_W
    else:
        d2, w2, h2, s2, l2, t2, gap = float(SCREEN_W * 2), 0, 0, 0.0, 0.0, 1e3, float(SCREEN_W)

    feats = [y, d1, h1, w1, speed, d2, h2, w2, t1, t2, gap, s1, l1, s2, l2, *traits]

    # Robust gegen verschiedene NEAT-Konfigurationen:
    try:
        n = len(getattr(net, "input_nodes", [])) or DEFAULT_INPUTS
    except Exception:
        n = DEFAULT_INPUTS

    return feats[:n] if len(feats) >= n else feats + [0.0] * (n - len(feats))

# ================== Core Loop ==================
def main_game(genomes, config):
    clock = pygame.time.Clock()
    epoch, points, spd, xbg = 1, 0, BASE_SPEED, 0
    obstacles, spawner = [], Spawner()

    # Population (RNN + Traits)
    nets, dinos, ge, traits = [], [], [], []
    for _, g in genomes:
        # leichte Zufallsinitialisierung der vorhandenen Verbindungen
        for c in g.connections.values():
            c.weight = random.uniform(-1, 1) * random.choice([0.5, 1.0, 1.5, 2.0])
        net = neat.nn.RecurrentNetwork.create(g, config)
        net.reset()
        nets.append(net); dinos.append(Dino()); g.fitness = 0.0; ge.append(g)
        traits.append([random.uniform(-1, 1) for _ in range(TRAITS_DIM)])

    best_ever, best_fit = None, -1e9

    def new_epoch():
        nonlocal nets, dinos, ge, traits, obstacles, spawner, points, epoch, spd, best_ever
        base = best_ever if best_ever is not None else (ge[0] if ge else None)
        if base is None:
            return
        muts = []
        for _ in range(max(10, len(ge) or 10)):
            g = copy.deepcopy(base)
            g.mutate(config.genome_config)
            for c in g.connections.values():
                c.weight += random.uniform(-0.25, 0.25)
            g.fitness = 0.0
            muts.append(g)
        nets, dinos, ge, traits = [], [], [], []
        for g in muts:
            net = neat.nn.RecurrentNetwork.create(g, config); net.reset()
            nets.append(net); dinos.append(Dino()); ge.append(g)
            traits.append([random.uniform(-1, 1) for _ in range(TRAITS_DIM)])
        obstacles.clear(); spawner = Spawner(); points = 0; epoch += 1; spd = BASE_SPEED

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                return -1  # vorzeitig beenden

        SCREEN.fill((255, 255, 255))
        spawner.maybe_spawn(obstacles, spd, epoch)

        o1 = obstacles[0] if obstacles else None
        o2 = obstacles[1] if len(obstacles) > 1 else None

        best_idx, best_now = None, -1e9
        i = 0
        while i < len(dinos):
            d = dinos[i]
            ge[i].fitness += 0.1
            inp = build_inputs(nets[i], d, o1, o2, spd, traits[i])
            if nets[i].activate(inp)[0] > 0.5 and d.rect.y == d.Y:
                d.jump = True; d.run = False
            d.update(); d.draw()
            if ge[i].fitness > best_now:
                best_now = ge[i].fitness; best_idx = i
            i += 1

        j = 0
        while j < len(obstacles):
            o = obstacles[j]; o.draw()
            if o.upd(spd):
                obstacles.pop(j); continue
            j += 1

        best_died = False
        i = 0
        while i < len(dinos):
            d = dinos[i]
            dead = any(d.rect.colliderect(o.rect) for o in obstacles)
            if dead:
                ge[i].fitness -= 10
                if i == best_idx:
                    best_died = True
                dinos.pop(i); ge.pop(i); nets.pop(i); traits.pop(i); continue
            i += 1

        if best_idx is not None and len(ge) > 0 and ge[best_idx].fitness > best_fit:
            best_fit = ge[best_idx].fitness
            best_ever = copy.deepcopy(ge[best_idx])

        if best_died or len(dinos) == 0:
            new_epoch()

        points += 1
        if points % SPEED_STEP_POINTS == 0 and spd < MAX_SPEED:
            spd += 1

        hud(points, epoch, spd)
        xbg = scroll_bg(xbg, spd)

        pygame.display.update()
        clock.tick(FPS)

# ================== NEAT Wrapper & Menü ==================
NEAT_CONFIG_TEXT = """# Auto-generated NEAT config (recurrent, 18 inputs, 1 output)
[NEAT]
fitness_criterion     = max
fitness_threshold     = 5000.0
pop_size              = 50
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid

# aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 5.0
bias_min_value          = -5.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.2
conn_delete_prob        = 0.1

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

# structure options
feed_forward            = False
initial_connection      = full_direct
node_add_prob           = 0.05
node_delete_prob        = 0.03

# network parameters
num_hidden              = 0
num_inputs              = 18
num_outputs             = 1

# output activation
output_activation       = sigmoid

# weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 8.0
weight_min_value        = -8.0
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func    = max
max_stagnation          = 15
species_elitism         = 2

[DefaultReproduction]
elitism                 = 2
survival_threshold      = 0.2
"""

def ensure_neat_config(filename="config-feedforward.txt"):
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(NEAT_CONFIG_TEXT)
        print(f"[Info] NEAT-Config erzeugt: {path}")
    return path

def run_neat(cfg_path):
    cfg = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        cfg_path,
    )
    pop = neat.Population(cfg)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    print("\nDrücke ESC während des Trainings, um vorzeitig zu beenden.\n")

    def eval(genomes, config):
        return True if main_game(genomes, config) == -1 else False

    # Mehrere Generationen iterativ aufrufen, damit ESC greift
    for _ in range(50):
        stop = pop.run(eval, 1)
        if stop:
            print("\nTraining beendet.\n")
            break

def menu():
    running = True
    big = pygame.font.Font('freesansbold.ttf', 30)
    while running:
        SCREEN.fill((255, 255, 255))
        for i, txt in enumerate(("T-Rex Runner NEAT (RNN)", "1. Train", "2. Quit")):
            SCREEN.blit(big.render(txt, True, (0, 0, 0)), (SCREEN_W // 2 - 200, 120 + i * 80))
        pygame.display.update()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_1:
                    cfg = ensure_neat_config()
                    run_neat(cfg)
                if e.key == pygame.K_2:
                    running = False
    pygame.quit()

# ================== Main ==================
if __name__ == "__main__":
    menu()
