import ast

with open('src/app.py') as f:
    tree = ast.parse(f.read())

functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

required = ['render_sidebar', 'render_overview_tab', 'render_jmes_tab', 'main']
missing = [f for f in required if f not in functions]
assert not missing, f"Missing functions: {missing}"

print("Step 5: App structure verified")
print(f"  Found {len(functions)} functions")
print(f"  Required functions: {', '.join(required)}")
