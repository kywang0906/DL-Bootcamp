import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    
    def slope(self):
        """Slope = (y2 - y1) / (x2 - x1)"""
        return (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)

    def isParallel(self, other):
        """Two lines are parallel if their slopes are equal"""
        return self.slope() == other.slope()
    
    def isPerpendicular(self, other):
        """Two lines are perpendicular if the product of their slopes is -1"""
        return self.slope() * other.slope() == -1

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    
    def area(self):
        return math.pi * (self.radius ** 2)

    def intersect(self, other):
        """
        Two circles intersect if the distance between their centers is between
        the sum and the absolute difference of their radii
        """
        return (self.center.distance(other.center) <= (self.radius + other.radius)) and \
                (self.center.distance(other.center) >= abs(self.radius - other.radius))
    
class Polygon:
    def __init__(self, points):
        self.points = points
    
    def perimeter(self):
        """Perimeter is the sum of the distances between consecutive points"""
        perimeter = 0
        for i in range(len(self.points)-1):
            perimeter += self.points[i].distance(self.points[(i + 1)])
        perimeter += self.points[-1].distance(self.points[0])
        return perimeter

class Enemy(Point):
    def __init__(self, x, y, dx, dy):
        super().__init__(x, y)
        self.points = 10
        self.dx = dx
        self.dy = dy
    
    def move(self):
        if self.isAlive():
            self.x += self.dx
            self.y += self.dy
    
    def losePoints(self, attacks):
        self.points -= attacks
    
    def isAlive(self):
        return self.points > 0

class BasicTower(Point):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.attackRange = 2
        self.attackPoints = 1
    
    def attack(self, enemies):
        """For each enemy in range, attack it"""
        for enemy in enemies:
            if enemy.isAlive() and self.distance(enemy) <= self.attackRange:
                enemy.losePoints(self.attackPoints)

class AdvancedTower(BasicTower):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.attackRange = 4
        self.attackPoints = 2

def task1():
    lineA = Line(Point(-6, 1), Point(2, 4))
    lineB = Line(Point(-6, -1), Point(2, 2))
    lineC = Line(Point(-1, 6), Point(-4, -4))
    circleA = Circle(Point(6, 3), 2)
    circleB = Circle(Point(8, 1), 1)
    polygonA = Polygon([Point(2, 0), Point(5, -1), Point(4, -4), Point(-1, -2)])
    print("Are Line A and Line B parallel?", lineA.isParallel(lineB))
    print("Are Line C and Line A perpendicular?", lineC.isPerpendicular(lineA))
    print("Print the area of Circle A.", circleA.area())
    print("Do Circle A and Circle B intersect?", circleA.intersect(circleB))
    print("Print the perimeter of Polygon A.", polygonA.perimeter())

def task2():
    E1 = Enemy(-10, 2, 2, -1)
    E2 = Enemy(-8, 0, 3, 1)
    E3 = Enemy(-9, -1, 3, 0)
    T1 = BasicTower(-3, 2)
    T2 = BasicTower(-1, -2)
    T3 = BasicTower(4, 2)
    T4 = BasicTower(7, 0)
    A1 = AdvancedTower(1, 1)
    A2 = AdvancedTower(4, -3)
    enemies = [E1, E2, E3]

    for i in range(10):
        # Enemies move forward
        for enemy in enemies:
            enemy.move()
        # Towers Attack
        for tower in [T1, T2, T3, T4, A1, A2]:
            tower.attack(enemies)

    print(f"E1 {E1.x} {E1.y} {E1.points}")
    print(f"E2 {E2.x} {E2.y} {E2.points}")
    print(f"E3 {E3.x} {E3.y} {E3.points}")


def main():
    task1()
    print("="*20)
    task2()

if __name__ == "__main__":
    main()