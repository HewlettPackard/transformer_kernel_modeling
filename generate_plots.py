import subprocess
import os
import math

def generate_parameter_table(kernel_parameters, output_dir=".", filename="kernel_parameters",
                             show_full_range=False, show_decimals=False, use_math_interval=True):
    """
    Generate a compact LaTeX table from a dictionary of kernel parameters,
    including cumulative search space size.

    Args:
        kernel_parameters: Dictionary with parameter names as keys and lists of values as values
        output_dir: Directory to save the output files (default: current directory)
        filename: Base name for the output files (default: kernel_parameters)
        show_full_range: Whether to show all values or just min/max (default: False)
        show_decimals: Whether to show decimal equivalents for powers of 2 (default: False)
        use_math_interval: Whether to use mathematical interval notation [low, high] (default: True)

    Returns:
        Path to the generated PDF file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate the cumulative search space sizes
    param_names = list(kernel_parameters.keys())
    cumulative_sizes = []
    cumulative_product = 1

    for param in param_names:
        cumulative_product *= len(kernel_parameters[param])
        cumulative_sizes.append(cumulative_product)

    # Start LaTeX document - with tighter borders and better table fitting
    latex_lines = [
        "\\documentclass[border=5pt,varwidth]{standalone}",
        "\\usepackage{booktabs}",
        "\\usepackage{array}",
        "\\usepackage{amsmath}",
        "\\usepackage{siunitx}",  # For scientific notation
        "\\usepackage[table]{xcolor}",
        "\\begin{document}",
        "",
        "\\begin{tabular}{llr}",  # Added third column for search space size
        "\\toprule",
        "\\textbf{Parameter} & \\textbf{Values} & \\textbf{Cummulative Space} \\\\",
        "\\midrule"
    ]

    # Function to format large numbers in scientific notation
    def format_scientific(number):
        if number < 1000:
            return str(number)  # Small numbers as integers
        exponent = math.floor(math.log10(number))
        mantissa = number / (10 ** exponent)
        # Fixed - using proper f-string syntax
        return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"

    # Add rows for each parameter
    for i, param in enumerate(param_names):
        values = sorted(kernel_parameters[param])

        # Check if all values are powers of 2
        is_power_of_2 = all(val & (val-1) == 0 and val > 0 for val in values)

        # Determine which values to show based on show_full_range
        if not show_full_range and len(values) > 2:
            display_values = [min(values), max(values)]
        else:
            display_values = values

        if is_power_of_2:
            # Get exponents for the values to display
            exponents = [int.bit_length(val) - 1 for val in display_values]

            if show_full_range:
                # Show all values (with compact formatting)
                if show_decimals:
                    formatted_values = ", ".join([f"$2^{{{exp}}}\\!=\\!{2**exp}$" for exp in exponents])
                else:
                    formatted_values = ", ".join([f"$2^{{{exp}}}$" for exp in exponents])
            else:
                # Show min/max only
                if len(exponents) == 1:
                    # Only one value
                    if show_decimals:
                        formatted_values = f"$2^{{{exponents[0]}}}\\!=\\!{2**exponents[0]}$"
                    else:
                        formatted_values = f"$2^{{{exponents[0]}}}$"
                else:
                    # Min and max values
                    if use_math_interval:
                        if show_decimals:
                            formatted_values = f"$[2^{{{exponents[0]}}}\\!=\\!{2**exponents[0]}," + \
                                              f"2^{{{exponents[-1]}}}\\!=\\!{2**exponents[-1]}]$"
                        else:
                            formatted_values = f"$[2^{{{exponents[0]}}},2^{{{exponents[-1]}}}]$"
                    else:
                        if show_decimals:
                            formatted_values = f"$2^{{{exponents[0]}}}\\!=\\!{2**exponents[0]}$ to " + \
                                              f"$2^{{{exponents[-1]}}}\\!=\\!{2**exponents[-1]}$"
                        else:
                            formatted_values = f"$2^{{{exponents[0]}}}$ to $2^{{{exponents[-1]}}}$"
        else:
            # For non-powers of 2
            if not show_full_range and len(display_values) > 1:
                if use_math_interval:
                    formatted_values = f"$[{display_values[0]},{display_values[-1]}]$"
                else:
                    formatted_values = f"{display_values[0]} to {display_values[-1]}"
            else:
                formatted_values = ", ".join([str(v) for v in display_values])

        # Escape underscores in parameter names
        param_escaped = param.replace("_", "\\_")

        # Format cumulative search space size
        search_space = format_scientific(cumulative_sizes[i])

        # Add row to table with search space size
        latex_lines.append(f"{param_escaped} & {formatted_values} & {search_space} \\\\")

    # Finish the table and document
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "\\end{document}"
    ])

    # Write LaTeX content to file
    tex_file = os.path.join(output_dir, f"{filename}.tex")
    with open(tex_file, "w") as f:
        f.write("\n".join(latex_lines))

    # Compile LaTeX to PDF
    try:
        subprocess.run(["pdflatex", "-output-directory", output_dir, tex_file],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Clean up auxiliary files
        for ext in [".aux", ".log"]:
            aux_file = os.path.join(output_dir, f"{filename}{ext}")
            if os.path.exists(aux_file):
                os.remove(aux_file)

        print(f"PDF generated and saved to {os.path.join(output_dir, filename)}.pdf")
        return os.path.join(output_dir, f"{filename}.pdf")

    except subprocess.CalledProcessError as e:
        print(f"Error generating PDF: {e}")
        return None

# Example usage
kernel_parameters = {
    "batch": [2**i for i in range(0, 8)],  # 1 to 128
    "max_seq_length": [2**i for i in range(8, 15)],  # 256 to 16384
    "n_embed": [2**i for i in range(10, 15)],  # 1024 to 16384
    "n_head": [2**i for i in range(3, 8)],  # 8 to 128
    "tensor_parallelism": [2**i for i in range(0, 4)],  # 1 to 8
}

generate_parameter_table(kernel_parameters, "results/sc25/figures", "kernel_params")
