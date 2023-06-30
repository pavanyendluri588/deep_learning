import turtle
import time
import datetime

# Set up the turtle screen
screen = turtle.Screen()
screen.setup(width=1300, height=900)
screen.bgcolor("#333333")

# Set up the turtle
turtle.speed(0)
turtle.hideturtle()

# Define the draw heart function
def draw_heart(x, y, size, color):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    turtle.color(color)
    turtle.fillcolor(color)
    turtle.begin_fill()
    turtle.left(45)
    turtle.forward(size)
    turtle.circle(size/2, 180)
    turtle.right(90)
    turtle.circle(size/2, 180)
    turtle.forward(size)
    turtle.end_fill()

# Draw the hearts
colors = ["#FF4136", "#FF851B", "#FFDC00", "#2ECC40", "#0074D9", "#B10DC9"]
x_positions = [-450, -150, 150, 450]
y_positions = [-200, 200]
for i in range(len(x_positions)):
    for j in range(len(y_positions)):
        x = x_positions[i]
        y = y_positions[j]
        draw_heart(x, y, 100, colors[(i+j) % len(colors)])

# Animate the hearts and message
for i in range(300):
    turtle.clear()
    for j in range(len(x_positions)):
        for k in range(len(y_positions)):
            x = x_positions[j]
            y = y_positions[k]
            size = 100 + i * 5
            draw_heart(x, y, size, colors[(j+k+i) % len(colors)])
    turtle.penup()
    turtle.goto(0, -350)
    turtle.color("#FFFFFF")
    turtle.write("Happy Birthday!", align="center", font=("Arial", 80 + i, "bold"))
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    turtle.goto(0, -400)
    turtle.write(f"Current Time: {current_time}", align="center", font=("Arial", 20, "normal"))
    turtle.update()
    time.sleep(3)

# Hide the turtle and exit
turtle.hideturtle(8)
turtle.done()
