count = 1
for i in range(1,13):
    if i == 1 or i == 3 or i == 5 or i == 7 or i == 8 or i == 10 or i == 12:
        for j in range(1, 32):
            file = open("dates_leap.csv", "a+")

            file.write(str(i) + "/" + str(j) + "," + str(count) + "\n")

            file.close()
            count += 1
    if i == 4 or i == 6 or i == 9 or i == 11:
        for j in range(1, 31):
            file = open("dates_leap.csv", "a+")

            file.write(str(i) + "/" + str(j) + "," + str(count) + "\n")

            file.close()
            count += 1
    if i == 2:
        for j in range(1, 30):
            file = open("dates_leap.csv", "a+")

            file.write(str(i) + "/" + str(j) + "," + str(count) + "\n")

            file.close()
            count += 1