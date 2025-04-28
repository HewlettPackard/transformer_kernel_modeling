options(error = function() { quit(save = "no", status = 1) })
.libPaths("/usr/local/lib/R/site-library")
check_and_install_packages <- function(packages) {
  # Identify packages that are not installed
  missing_packages <- packages[!packages %in% installed.packages()[, "Package"]]

  # Install missing packages
  if (length(missing_packages) > 0) {
    install.packages(missing_packages)
  } else {
    message("All packages are already installed.")
  }
}

# Example usage
required_packages <- c("tidyverse", "RColorBrewer", "patchwork", "ggh4x", "ggtext",
                       "scales", "latex2exp")

check_and_install_packages(required_packages)
