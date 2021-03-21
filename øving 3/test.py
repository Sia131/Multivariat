from prettytable import PrettyTable



x = PrettyTable()

x.field_names = [" ", "City name", "Area", "Population", "Annual Rainfall"]
x.add_column(" ",[1, 2 ,3])
print(x)