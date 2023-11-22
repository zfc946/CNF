import torch
import re

# Evaluate the value of Fy(x) where F is a linear operator of y(x) expressed by a latex string
def compute_op(latex_str, y, x):
	'''
	Example: - f_{x_0x_1} + x_3
	Properties of LaTeX input:
		1. Include space between operator and operands
		2. Multiple differentation is considered under a single operand
		3. No space in subscript for differentiation
		4. Unary operator (+/-) at the beginning is optional
	'''
	def parse_latex(latex_str, y, x):
		latex_str = latex_str.strip(' ')
		if latex_str[0] not in ('+', '-'):
			latex_str = '+ ' + latex_str

		# TODO: Multiplication/division? We can use postfix expressions
		operators = latex_str.split(' ')[::2]
		operands = latex_str.split(' ')[1::2]
		assert all([op in ('+', '-') for op in operators]), 'Only addition and subtraction are currently supported'

		result = 0
		for operator, operand in zip(operators, operands):
			if operator == '+':
				result += process_operand(operand, y, x)
			elif operator == '-':
				result -= process_operand(operand, y, x)
		return result

	def process_operand(op, y, x):
		# Check for numeraic constants
		_, k, op = re.split('^(\d*\.?\d*)', op)
		k = 1 if k == '' else float(k)

		if op.startswith('f_'):
			# Perform differentiation (use loop for higher-order terms)
			# TODO: Cache intermediate results? Only useful for very complicated expressions
			derivates = re.search('^f_\{([\{x_\d\}]+)\}$', op).group(1).split('x_')
			assert len(derivates) > 1, f'Invalid operand: {op}'
			for d in derivates[1:]:
				try:
					d = re.search('^\{(\d+)\}$', d).group(1)
				except AttributeError:
					pass
				y = compute_partial_differentiation(y, x, int(d))
		elif op.startswith('x_'):
			# No differentiation, just retrieve the element
			try:
				d = re.search('^x_\{(\d+)\}$').group(1)
			except AttributeError:
				d = re.search('^x_(\d+)$').group(1)
			y = x[int(d)]
		elif op == 'f':
			y = y
		else:
			RuntimeError(f'Unknown operand: {op}')
		return k*y

	def compute_partial_differentiation(y, x, d):
		# compute the partial differentiation of y with respect to x_d
		partial = torch.zeros_like(y)
		for i in range(y.shape[-1]):
			partial[..., i] = torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True, retain_graph=True, allow_unused=True)[0][..., d]
		return partial

	return parse_latex(latex_str, y, x)