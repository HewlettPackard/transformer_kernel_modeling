options(error = function() { quit(save = "no", status = 1) })
options(tidyverse.quiet = TRUE)
options(readr.show_progress = FALSE)
options(readr.show_col_types = FALSE)

library(tidyverse)

data_dir <- "results/sc25/raw_kernel_profiling_data/a100_mean_factorial_expanded"

scaled_csv_files <- list.files(path = data_dir,
                               pattern = "scaled_experiments\\.csv$",
                               full.names = TRUE,
                               recursive = TRUE)

scaled_combined_df <- scaled_csv_files |>
    lapply(read_csv) |>
    bind_rows() |>
    write_csv("results/sc25/scaled_kernel_profiling_data.csv")

normalized_csv_files <- list.files(path = data_dir,
                                   pattern = "normalized_experiments\\.csv$",
                                   full.names = TRUE,
                                   recursive = TRUE)

normalized_combined_df <- normalized_csv_files |>
    lapply(read_csv) |>
    bind_rows() |>
    write_csv("results/sc25/normalized_kernel_profiling_data.csv")

library(tidyverse)

df <- read_csv("results/sc25/scaled_kernel_profiling_data.csv",
               show_col_types = FALSE) |>
    filter(!is.na(runtime_s)) |>
    mutate(seq_length_filled = coalesce(seq_length,
                                        max_seq_length)) |>
    mutate(seq_length = seq_length_filled) |>
    select(batch,
           seq_length,
           n_head,
           n_embed,
           tensor_parallelism,
           kernel,
           gpu_name,
           runtime_s) |>
    mutate(dot_color = batch) |>
    pivot_longer(cols = -c("kernel",
                           "gpu_name",
                           "dot_color",
                           "runtime_s"),
                 names_to = "parameter",
                 values_to = "value") |>
    filter(!is.na(runtime_s))

library(tidyverse)
library(RColorBrewer)
library(patchwork)
library(ggh4x)
library(ggtext)

legend_text_size <- 15
legend_title_size <- 15
legend_key_size <- 1.5
legend_bar_width <- 30
legend_bar_height <- 0.25
plot_title_size <- 15
axis_title_size <- 15
plot_base_size <- 16
strip_text_size <- 15

kernel_values <- unique(df$kernel)
plots_list <- list()

color_palette <- "Spectral"

for (k in kernel_values) {
    kernel_df <- df %>%
        filter(kernel == k) %>%
        filter(!is.na(value))

    legend_pos <- if(k == tail(kernel_values, 1)) "bottom" else "none"

    p <- ggplot(kernel_df, aes(x = value, y = runtime_s, color = dot_color)) +
        geom_jitter(width = 0.35, height = 0, size = 1.1, alpha = 0.9) +
        geom_smooth(data = kernel_df |> filter(dot_color %in% c(1, 16, 128)),
                    aes(color = dot_color,
                        group = dot_color),
                    se = TRUE,
                    alpha = 0.4,
                    method = "lm",
                    formula = y ~ x) +
        facet_grid2(
            . ~ parameter,
            scales = "free"
        ) +
        labs(x = paste0("Parameter values for ", k),
             y = "Time (s)") +
        scale_color_distiller(name = "Batch", palette = color_palette, trans = "log2") +
        scale_y_log10(labels = function(x) {
            exponents <- round(log10(x), 1)
            labels <- ifelse(exponents < 0,
                             paste0("10^{-", "~", abs(exponents), "}"),
                             paste0("10^{", exponents, "}"))
            parse(text = labels)
        }, n.breaks = 4) +
        scale_x_continuous(
            trans = "log2",
            n.breaks = 4,
            labels = function(x) parse(text = paste0("2^{", round(log2(x), 1), "}"))
        ) +
        guides(color = guide_colorbar(
                   title.position = "top",
                   barwidth = unit(legend_bar_width, "cm"),
                   barheight = unit(legend_bar_height, "cm"),
                   ticks = TRUE,
                   frame.colour = "black",
                   frame.linewidth = 0.0
               )) +
        theme_bw(base_size = plot_base_size) +
        theme(
            strip.background = element_rect(fill = "white", color = "gray80"),
            strip.text = element_text(size = strip_text_size),
            legend.position = legend_pos,
            legend.box.margin = margin(l = -850, unit = "pt"),
            legend.box = "horizontal",
            legend.text = element_text(size = legend_text_size),
            legend.title = element_text(size = legend_title_size),
            legend.key.size = unit(legend_key_size, "lines"),
            plot.title = element_text(size = plot_title_size, hjust = 0),
            axis.title = element_text(size = axis_title_size)
        )

    plots_list[[k]] <- p
}

#combined_plot <- wrap_plots(plots_list, ncol = 2)
combined_plot <- (plots_list[[1]] / plots_list[[2]]) / (plots_list[[3]] + plots_list[[4]])
ggsave("results/sc25/figures/combined_kernel_plots.png",
       combined_plot, width = 20, height = 11, dpi = 600)

library(tidyverse)
library(patchwork)
library(RColorBrewer)
library(ggtext)
library(latex2exp)

options(tidyverse.quiet = TRUE)
options(readr.show_progress = FALSE)
options(readr.show_col_types = FALSE)

find_all_files <- function(path, pattern, recursive = TRUE) {
    list.files(path = path,
               pattern = pattern,
               recursive = recursive,
               full.names = TRUE)
}

load_files_in_batches <- function(files, batch_size = 500) {
    batch_no <- (seq_along(files) - 1) %/% batch_size

    split(files, batch_no) |>
        purrr::map_df(function(x) {
            read_csv(x,
                    id = "origin_file",
                    show_col_types = FALSE,
                    name_repair = "minimal")
        })
}

format_simulation_data <- function(df_sim_full) {
    df_sim_full |>
        select(origin_file, `workload finished at`) |>
        drop_na() |>
        separate_wider_regex(origin_file,
                             c("results/sc25/astra_sim_measurements/",
                               gpu = ".*?", "/",
                               type = ".*?", "/", gpus = "\\d+",
                               "_gpu_", ubatch = "\\d", "_ubatch_",
                               coef = "\\d+(?:\\.\\d+)?|.*?", "_coef_",
                               rep = "\\d+", ".*"), too_few = "align_start") |>
        group_by(type, gpus, ubatch, coef) |>
        summarize(total_time_ms = mean(`workload finished at` / 1e6),
                  total_time_ms_ci = (1.97 * sd(`workload finished at` / 1e6)) / sqrt(n())) |>
        ungroup() |>
        mutate(total_time_ms_ci = ifelse(is.na(total_time_ms_ci), 0.0, total_time_ms_ci))
}

load_and_format_real_data <- function(path_pattern, origin_file_pattern = ".*?llama-7B_NVIDIA-A100-SXM4-80GB_gpu-") {
    read_csv(list.files(path = path_pattern$path,
                        pattern = path_pattern$pattern,
                        recursive = TRUE,
                        full.names = TRUE),
             id = "origin_file") |>
        group_by(origin_file) |>
        slice(3:n()) |> # Drop two first runs
        summarize(total_time_ms = mean(Back + Fwd) / 1e3,
                  total_time_ms_mean = mean(Back + Fwd) / 1e3,
                  total_time_ms_ci = ((1.97 * sd(Back + Fwd) / sqrt(n()))) / 1e3) |>
        ungroup() |>
        select(origin_file, total_time_ms, total_time_ms_ci) |>
        separate_wider_regex(origin_file,
                             c(origin_file_pattern,
                               gpus = "\\d+", "_batch-",
                               ubatch = "\\d+", ".*")) |>
        mutate(type = "real")
}

adjust_data_and_save <- function(df_sim, df_real, output_path) {
    df <- bind_rows(df_sim, df_real)

    # Apply adjustments
    df[df$type == "roofline", "total_time_ms"] <- df[df$type == "roofline", "total_time_ms"] * 8

    write_csv(df, output_path)
    return(df)
}

load_performance_data <- function(file_path) {
    read_csv(file_path)
}

create_reference_dataframe <- function(df) {
    df |>
        filter(type == "real") |>
        select(gpus, ubatch, total_time_ms, ref_time = total_time_ms)
}

calculate_performance_ratios <- function(df, ref_df) {
    df |>
        left_join(ref_df, by = c("gpus", "ubatch")) |>
        group_by(type, gpus, ubatch) |>
        mutate(ratio_time_ms = total_time_ms / ref_time) |>
        ungroup() |>
        mutate(abs_diff_ratio = abs(ratio_time_ms - 1)) |>
        group_by(type, gpus, ubatch) |>
        filter(abs_diff_ratio == min(abs_diff_ratio)) |>
        ungroup()
}

save_and_print_ratios <- function(ratio_df, output_file) {
    ratio_df |> write_csv(output_file)
    #ratio_df |> print(n = Inf)
    return(ratio_df)
}

prepare_model_data <- function(ratio_df, type_filter = "model", ubatch_filter = 1) {
    ratio_df |>
        filter(type == type_filter,
               ubatch > ubatch_filter) |>
        select(gpus,
               ubatch,
               coef,
               total_time_ms,
               abs_diff_ratio) |>
        mutate_all(~ as.numeric(.))
}

fit_coefficient_model <- function(df_lm, formula_str = "coef ~ log(ubatch) + log(gpus) + I(1/ubatch) + I(1/gpus)") {
    model <- lm(data = df_lm, formula = formula_str)
    print(summary(model))
    return(model)
}

create_coefficient_heatmap <- function(input_file, output_file, grid_size = 64,
                                       max_ubatch = 64, max_gpus = 16,
                                       width = 10, height = 5) {
    # Get the data ready
    df <- read_csv(input_file)

    # Create the fitted models
    model_fit <- lm(coef ~ (I(1/ubatch) + I(1/gpus) + gpus),
                    data = filter(df, type == "model"))
    profiling_fit <- lm(coef ~ (I(1/ubatch) + I(1/gpus) + gpus),
                        data = filter(df, type == "profiling"))

    # Create a grid of values for prediction
    ubatch_range <- seq(min(df$ubatch), max_ubatch, length.out = grid_size)
    gpus_range <- seq(min(df$gpus), max_gpus, length.out = grid_size)
    prediction_grid <- expand.grid(ubatch = ubatch_range, gpus = gpus_range)

    # Make predictions
    prediction_grid$model_pred <- predict(model_fit, newdata = prediction_grid)
    prediction_grid$profiling_pred <- predict(profiling_fit, newdata = prediction_grid)
    prediction_grid$difference <- prediction_grid$model_pred - prediction_grid$profiling_pred

    # Convert to long format for easier plotting
    pred_long <- prediction_grid %>%
        pivot_longer(cols = c(model_pred, profiling_pred, difference),
                     names_to = "type",
                     values_to = "prediction")

    # Create heatmaps
    p <- ggplot(pred_long, aes(x = ubatch, y = gpus, fill = prediction)) +
        geom_tile() +
        scale_fill_gradientn(
            colors = rev(brewer.pal(11, "RdBu")),
            values = scales::rescale(c(min(pred_long$prediction), 1.0, max(pred_long$prediction))),
            limits = c(min(pred_long$prediction), max(pred_long$prediction))
        ) +
        facet_wrap(~ type, ncol = 3,
                   labeller = labeller(type = c("model_pred" = "Model",
                                               "profiling_pred" = "Profiling",
                                               "difference" = "Difference"))) +
        labs(title = "Predictions from the Fitted Models",
             x = "ubatch", y = "gpus", fill = "Coefficient") +
        theme_minimal() +
        theme(panel.grid.major = element_blank(),
              panel.grid.minor = element_blank())

    # Save the plot
    ggsave(output_file, p, width = width, height = height)

    return()
}

create_coefficient_trend_plots <- function(input_file, output_file,
                                          grid_size = 32, max_ubatch = 8, max_gpus = 128,
                                          output_width = 12, output_height = 10) {
    # Visual parameters
    line_alpha <- 0.5
    line_width <- 1.3
    point_alpha <- 0.5
    point_size <- 1.5
    color_palette <- "Spectral"
    reference_line_color <- "darkgray"
    reference_line_type <- "dashed"
    reference_line_value <- 1.0
    plot_base_size <- 12

    # Text parameters
    main_title <- "Coefficient Trends for Simulation Methods"
    main_subtitle <- "Values closer to 1.0 indicate better alignment with ground truth"
    gpus_plot_title <- "Coefficient by GPU Count"
    gpus_plot_subtitle_base <- "Each line represents a different batch size"
    ubatch_plot_title <- "Coefficient by Batch Size"
    ubatch_plot_subtitle <- "Each line represents a different GPU count"
    x_label_gpus <- "GPUs"
    x_label_ubatch <- "Batch Size"
    y_label <- "Predicted Coefficient"

    # Model parameters
    simpler_formula_template <- "coef = %s + (%s)*1/ubatch + (%s)*1/gpus, Adj R² = %s"
    model_prediction_columns <- c("model_pred", "profiling_pred")
    model_type_levels <- c("model_pred", "profiling_pred")
    model_type_labels <- c("Model", "Profiling")

    # Get the data ready
    df <- read_csv(input_file)

    # Create the fitted models
    model_fit <- lm(coef ~ (I(1/ubatch) + I(1/gpus)),
                    data = filter(df, type == "model"))
    profiling_fit <- lm(coef ~ (I(1/ubatch) + I(1/gpus)),
                        data = filter(df, type == "profiling"))

    # Format model formulas
    model_formula <- sprintf(simpler_formula_template,
                             round(coef(model_fit)[1], 3),
                             round(coef(model_fit)[2], 3),
                             round(coef(model_fit)[3], 3),
                             round(summary(model_fit)$adj.r.squared, 3))
    prof_formula <- sprintf(simpler_formula_template,
                            round(coef(profiling_fit)[1], 3),
                            round(coef(profiling_fit)[2], 3),
                            round(coef(profiling_fit)[3], 3),
                            round(summary(profiling_fit)$adj.r.squared, 3))

    # Create combined title with formulas
    gpus_plot_subtitle <- paste0(gpus_plot_subtitle_base, " \nModel: ", model_formula,
                                 "\nProfiler: ", prof_formula)

    # Create a dense grid for predictions
    ubatch_range <- seq(min(df$ubatch), max_ubatch, length.out = grid_size)
    gpus_range <- seq(min(df$gpus), max_gpus, length.out = grid_size)
    prediction_grid <- expand.grid(ubatch = ubatch_range, gpus = gpus_range)

    # Make predictions
    prediction_grid$model_pred <- predict(model_fit, newdata = prediction_grid)
    prediction_grid$profiling_pred <- predict(profiling_fit, newdata = prediction_grid)

    pred_long <- prediction_grid %>%
        pivot_longer(cols = model_prediction_columns,
                     names_to = "source",
                     values_to = "prediction") %>%
        mutate(source = factor(source,
                               levels = model_type_levels,
                               labels = model_type_labels))

    # Create GPUs plot
    gpus_plot <- ggplot(pred_long, aes(x = gpus, y = prediction)) +
        geom_line(aes(group = interaction(source, ubatch), color = ubatch),
                  alpha = line_alpha, linewidth = line_width) +
        geom_point(aes(color = ubatch),
                   alpha = point_alpha, size = point_size) +
        geom_hline(yintercept = reference_line_value,
                   linetype = reference_line_type,
                   color = reference_line_color) +
        geom_jitter(data = df |>
                        filter(type == "model" | type == "profiling") |>
                        mutate(source = factor(type,
                                               levels = c("model", "profiling"),
                                               labels = model_type_labels)),
                    aes(x = gpus,
                        y = as.numeric(coef)),
                    color = "black",
                    width = 0,
                    height = 0,
                    alpha = 0.7,
                    size = 1.5,
                    show.legend = FALSE) +
        facet_wrap(~ source) +
        scale_color_distiller(palette = color_palette) +
        labs(
            title = gpus_plot_title,
            subtitle = gpus_plot_subtitle,
            x = x_label_gpus,
            y = y_label
        ) +
        theme_bw(base_size = plot_base_size) +
        theme(
            legend.position = "right",
            strip.background = element_rect(fill = "white")
        )

    # Create ubatch plot
    ubatch_plot <- ggplot(pred_long, aes(x = ubatch, y = prediction)) +
        geom_line(aes(group = interaction(source, gpus), color = gpus),
                  alpha = line_alpha, linewidth = line_width) +
        geom_point(aes(color = gpus),
                   alpha = point_alpha, size = point_size) +
        geom_hline(yintercept = reference_line_value,
                   linetype = reference_line_type,
                   color = reference_line_color) +
        geom_jitter(data = df |>
                        filter(type == "model" | type == "profiling") |>
                        mutate(source = factor(type,
                                               levels = c("model", "profiling"),
                                               labels = model_type_labels)),
                    aes(x = ubatch,
                        y = as.numeric(coef)),
                    color = "black",
                    width = 0,
                    height = 0,
                    alpha = 0.7,
                    size = 1.5,
                    show.legend = FALSE) +
        facet_wrap(~ source) +
        scale_color_distiller(palette = color_palette) +
        labs(
            title = ubatch_plot_title,
            subtitle = ubatch_plot_subtitle,
            x = x_label_ubatch,
            y = y_label
        ) +
        theme_bw(base_size = plot_base_size) +
        theme(
            legend.position = "right",
            strip.background = element_rect(fill = "white")
        )

    # Combine plots and save
    final_plot <- gpus_plot / ubatch_plot +
        plot_annotation(
            title = main_title,
            subtitle = main_subtitle,
            theme = theme(
                plot.title = element_text(face = "bold", size = 14),
                plot.subtitle = element_text(face = "italic", size = 12)
            )
        )

    ggsave(output_file, final_plot, width = output_width, height = output_height)

    return()
}

create_performance_comparison_plot <- function(input_file = "results/sc25/simulation_results/performance_ratios_A100_factorial_expanded.csv",
                                               output_file = "results/sc25/figures/model_profiling_comparison_A100_factorial_expanded.pdf",
                                               width = 12,
                                               height = 8) {
    library(tidyverse)

    df <- read_csv(input_file)

    ground_truth <- df %>%
        filter(type == "real") %>%
        select(gpus, ubatch, reference = total_time_ms, total_time_ms_ci)

    plot_data <- df %>%
        filter(type %in% c("roofline", "profiling", "model")) %>%
        left_join(ground_truth, by = c("gpus", "ubatch", "total_time_ms_ci")) %>%
        select(ubatch, gpus, type, total_time_ms, total_time_ms_ci, reference) %>%
        mutate(
            ubatch = factor(ubatch,
                            levels = c("1", "2", "4", "8"),
                            labels = c("Batch 1", "Batch 2", "Batch 4", "Batch 8")),
            gpus = factor(gpus,
                          levels = c("1", "2", "4", "8", "16",
                                     "32", "64", "128"),
                          labels = c("1 GPU", "2 GPUs", "4 GPUs", "8 GPUs",
                                     "16 GPUs", "32 GPUs", "64 GPUs", "128 GPUs")),
            type = factor(type,
                          levels = c("roofline", "profiling", "model"),
                          labels = c("roofline", "profiler", "model"))
        )

    ground_truth_formatted <- ground_truth %>%
        mutate(
            ubatch = factor(ubatch,
                            levels = c("1", "2", "4", "8"),
                            labels = c("Batch 1", "Batch 2", "Batch 4", "Batch 8")),
            gpus = factor(gpus,
                          levels = c("1", "2", "4", "8",
                                     "16", "32", "64", "128"),
                          labels = c("1 GPU", "2 GPUs", "4 GPUs", "8 GPUs",
                                     "16 GPUs", "32 GPUs", "64 GPUs", "128 GPUs")),
            )

    p <- ggplot() +
        geom_rect(
            data = ground_truth_formatted,
            aes(
                xmin = -Inf,
                xmax = Inf,
                ymin = reference - total_time_ms_ci,
                ymax = reference + total_time_ms_ci
            ),
            fill = "grey",
            alpha = 0.6,
            inherit.aes = FALSE
        ) +
        geom_crossbar(data = plot_data, aes(x = type,
                                            y = total_time_ms,
                                            ymin = total_time_ms - total_time_ms_ci,
                                            ymax = total_time_ms + total_time_ms_ci,
                                            color = type,
                                            fill = type),
                      alpha = 0.4,
                      width = 0.2) +
        geom_hline(data = ground_truth_formatted,
                   aes(yintercept = reference),
                   linetype = "dashed",
                   color = "black") +
        geom_text(data = ground_truth_formatted,
                  aes(x = 1.1,
                      y = reference,
                      label = paste0("Ground Truth: ", round(reference, 2))),
                  vjust = -0.5, size = 3, color = "black") +
        geom_text(data = plot_data,
                  aes(x = type,
                      y = total_time_ms + total_time_ms_ci,
                      color = type,
                      label = round(total_time_ms, 2)),
                  vjust = -2.2, size = 3) +
        scale_color_brewer(palette = "Dark2") +
        scale_fill_brewer(palette = "Dark2") +
        guides(color = guide_legend(label = FALSE)) +
        facet_grid(ubatch ~ gpus, scales = "free_y") +
        labs(
            x = "Simulation Method",
            y = "Time (s)",
            title = "Comparison of Performance Simulation Methods"
        ) +
        scale_y_continuous(expand = expansion(mult = c(0.05, 0.25))) +
        theme_bw() +
        theme(
            legend.position = "none",
            strip.background = element_rect(fill = "white"),
            plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
        )

    dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)
    ggsave(output_file, p, width = width, height = height)

    return()
}

coefficient_trend_plots_publication <- function(input_file, output_file,
                                                grid_size = 32, max_ubatch = 2, max_gpus = 128,
                                                output_width = 12, output_height = 10) {
    legend_bar_height <- 6.5
    legend_bar_width <- 0.25
    line_alpha <- 0.5
    line_width <- 2
    point_alpha <- 0.5
    point_size <- 2
    color_palette <- "Set1"
    reference_line_color <- "black"
    reference_line_type <- "dashed"
    reference_line_value <- 1.0
    plot_base_size <- 18

    main_title <- "Coefficient Trends for Simulation Methods"
    main_subtitle <- "Values closer to 1.0 indicate better alignment with ground truth"
    gpus_plot_title <- "Coefficient by GPU Count"
    gpus_plot_subtitle_base <- "Each line represents a different batch size"
    ubatch_plot_title <- "Coefficient by Batch Size"
    ubatch_plot_subtitle <- "Each line represents a different GPU count"
    x_label_gpus <- "GPUs"
    x_label_ubatch <- "Batch Size"
    y_label <- "Correction Coefficient"

    # simpler_formula_template <- "$\\approx %.2f + \\left(%.2f log(ubatch)\\right) + \\left(%.2f log(gpus)\\right) + \\left(\\frac{%.2f}{ubatch}\\right) + \\left(\\frac{%.2f}{gpus}\\right)\\,, R^2=%.2f$"
    simpler_formula_template <- "$\\approx %.2f + %.2f \\cdot log(ubatch) + %.2f \\cdot log(gpus)$"
    R2_template <- "$R^2 = %.2f$"
    model_formula_template <- simpler_formula_template
    model_prediction_columns <- c("model_pred")
    model_type_levels <- c("model_pred")
    model_type_labels <- c("Model")

    df <- read_csv(input_file) |>
        mutate(ubatch = as.numeric(ubatch)) |>
        filter(ubatch <= 2,
               type == "model")
        #filter(type == "model")

    print(df |> distinct(ubatch))

    # model_fit <- lm(coef ~ (I(1/ubatch) + I(1/gpus)),
    #                 data = filter(df, type == "model"))

    model_fit <- lm(coef ~ (log(ubatch) + log(gpus)),
                    data = filter(df, type == "model"))

    print(summary(model_fit))

    model_formula <- sprintf(model_formula_template,
                             coef(model_fit)[1],
                             coef(model_fit)[2],
                             coef(model_fit)[3])

    model_r2 <- sprintf(R2_template,
                        summary(model_fit)$adj.r.squared)

    #ubatch_range <- seq(min(df$ubatch), max_ubatch, length.out = grid_size)
    #ubatch_range <- seq(min(df$ubatch), max_ubatch, length.out = grid_size)
    ubatch_range <- c(1, 2, 4)
    gpus_range <- c(1, 2, 4, 8, 16, 32, 64, 128)
    prediction_grid <- expand.grid(ubatch = ubatch_range, gpus = gpus_range)

    prediction_grid$model_pred <- predict(model_fit, newdata = prediction_grid)

    pred_long <- prediction_grid %>%
        pivot_longer(cols = model_prediction_columns,
                     names_to = "source",
                     values_to = "prediction") %>%
        mutate(source = factor(source,
                               levels = model_type_levels,
                               labels = model_type_labels))

    annotation_data <- data.frame(
        source = factor(c("Model"),
                        levels = model_type_labels),
        x = c(43),
        y = c(0.8),
        latex_label = c(model_formula)
    )

    r2_annotation_data <- data.frame(
        source = factor(c("Model"),
                        levels = model_type_labels),
        x = c(43),
        y = c(0.64),
        latex_label = c(model_r2)
    )

    annotation_data$latex_label <- sapply(annotation_data$latex_label,
                                          function (x) { TeX(x, output = "character") })

    r2_annotation_data$latex_label <- sapply(r2_annotation_data$latex_label,
                                             function (x) { TeX(x, output = "character") })

    gpus_plot <- ggplot(pred_long, aes(x = gpus, y = prediction)) +
        # geom_line(aes(group = interaction(source, ubatch), color = factor(ubatch)),
        #           alpha = line_alpha, linewidth = line_width) +
        # geom_point(aes(color = factor(ubatch)),
        #            alpha = point_alpha, size = point_size) +
        geom_hline(yintercept = reference_line_value,
                   linetype = reference_line_type,
                   color = reference_line_color) +
        geom_point(data = df |>
                       filter(type == "model") |>
                       mutate(source = factor(type,
                                              levels = c("model"),
                                              labels = model_type_labels)),
                   aes(x = gpus,
                       y = as.numeric(coef),
                       color = factor(ubatch)),
                   # shape = 18,
                   # width = 2,
                   # height = 0,
                   alpha = point_alpha,
                   size = point_size,
                   show.legend = TRUE) +
        geom_line(data = df |>
                      filter(type == "model") |>
                      mutate(source = factor(type,
                                             levels = c("model"),
                                             labels = model_type_labels)),
                  aes(x = gpus,
                      y = as.numeric(coef),
                      color = factor(ubatch)),
                  alpha = line_alpha,
                  size = line_width,
                  show.legend = TRUE) +
        # geom_label(data = annotation_data,
        #            aes(x = x,
        #                y = y,
        #                label = latex_label),
        #            hjust = 0.04,
        #            vjust = 0,
        #            size = 5.5,
        #            fill = "gray80",
        #            alpha = 0.5,
        #            label.padding = unit(0.4, "lines"),
        #            label.r = unit(0.2, "lines"),
        #            label.size = 0,
        #            parse = TRUE,
        #            inherit.aes = FALSE) +
        # geom_label(data = r2_annotation_data,
        #            aes(x = x,
        #                y = y,
        #                label = latex_label),
        #            hjust = 0.04,
        #            vjust = 0,
        #            size = 5.5,
        #            fill = "gray80",
        #            alpha = 0.5,
        #            label.padding = unit(0.4, "lines"),
        #            label.r = unit(0.2, "lines"),
        #            label.size = 0,
        #            parse = TRUE,
        #            inherit.aes = FALSE) +
        #scale_color_distiller(name = "Batch", palette = color_palette, trans = "log2") +
        scale_color_brewer(name = "Batch", palette = color_palette) +
        # scale_x_continuous(trans = "log2",
        #                    breaks = scales::breaks_log(n = 5)) +
        labs(
            x = x_label_gpus,
            y = y_label
        ) +
        theme_bw(base_size = plot_base_size) +
        theme(
            #legend.background = element_rect(fill = rgb(0, 0, 0, 0.13)),
            legend.title = element_text(size = 14),
            legend.direction = "horizontal",
            legend.position = c(0.8, 0.15),
            strip.background = element_rect(fill = "white"),
        )

    final_plot <- gpus_plot
    ggsave(output_file, final_plot, width = output_width, height = output_height)

    return()
}

ratio_df_table_publication <- function(ratio_df) {
    ratio_df |>
        filter(type == "model", ubatch <= 2) |>
        mutate(is_multinode = gpus > 8) |>
        group_by(ubatch, is_multinode) |>
        summarize(stat_coef = median(as.numeric(coef)))
}

performance_comparison_publication_A100 <- function(input_file = "results/sc25/simulation_results/performance_ratios_A100_factorial_expanded.csv",
                                                    output_file_batch_1_2_1_node = "results/sc25/figures/comparison_A100_batch_1_2_1_node.pdf",
                                                    output_file_batch_1_2_multi_node = "results/sc25/figures/comparison_A100_batch_1_2_multi_node.pdf",
                                                    output_file_batch_4_8 = "results/sc25/figures/comparison_A100_batch_4_8.pdf",
                                                    width = 12,
                                                    height = 8) {
    plot_base_size <- 14

    df <- read_csv(input_file)

    ground_truth <- df %>%
        filter(type == "real") %>%
        select(gpus, ubatch, reference = total_time_ms, total_time_ms_ci)

    plot_data <- df %>%
        filter(type %in% c("roofline", "profiling", "model")) %>%
        left_join(ground_truth, by = c("gpus", "ubatch", "total_time_ms_ci")) %>%
        select(ubatch, gpus, type, total_time_ms, total_time_ms_ci, reference) %>%
        mutate(
            ubatch = factor(ubatch,
                            levels = c("1", "2", "4", "8"),
                            labels = c("Batch 1", "Batch 2", "Batch 4", "Batch 8")),
            gpus = factor(gpus,
                          levels = c("1", "2", "4", "8", "16",
                                     "32", "64", "128"),
                          labels = c("1 GPU", "2 GPUs", "4 GPUs", "8 GPUs",
                                     "16 GPUs", "32 GPUs", "64 GPUs", "128 GPUs")),
            type = factor(type,
                          levels = c("roofline", "profiling", "model"),
                          labels = c("roofline", "profiler", "model"))
        )

    ground_truth_formatted <- ground_truth %>%
        mutate(
            ubatch = factor(ubatch,
                            levels = c("1", "2", "4", "8"),
                            labels = c("Batch 1", "Batch 2", "Batch 4", "Batch 8")),
            gpus = factor(gpus,
                          levels = c("1", "2", "4", "8",
                                     "16", "32", "64", "128"),
                          labels = c("1 GPU", "2 GPUs", "4 GPUs", "8 GPUs",
                                     "16 GPUs", "32 GPUs", "64 GPUs", "128 GPUs")),
            )

    p_ubatch_1_2_1_node <- ggplot() +
        geom_rect(
            data = ground_truth_formatted |>
                filter(ubatch == "Batch 1" | ubatch == "Batch 2",
                       gpus == "1 GPU" | gpus == "2 GPUs" | gpus == "4 GPUs" | gpus == "8 GPUs"),
            aes(
                xmin = -Inf,
                xmax = Inf,
                ymin = reference - total_time_ms_ci,
                ymax = reference + total_time_ms_ci
            ),
            fill = "grey",
            alpha = 0.6,
            inherit.aes = FALSE
        ) +
        geom_hline(data = ground_truth_formatted |>
                       filter(ubatch == "Batch 1" | ubatch == "Batch 2",
                              gpus == "1 GPU" | gpus == "2 GPUs" | gpus == "4 GPUs" | gpus == "8 GPUs"),
                   aes(yintercept = reference),
                   linetype = "dashed",
                   color = "black") +
        geom_crossbar(data = plot_data |>
                          filter(ubatch == "Batch 1" | ubatch == "Batch 2",
                                 gpus == "1 GPU" | gpus == "2 GPUs" | gpus == "4 GPUs" | gpus == "8 GPUs"),
                      aes(x = type,
                          y = total_time_ms,
                          ymin = total_time_ms - total_time_ms_ci,
                          ymax = total_time_ms + total_time_ms_ci,
                          color = type,
                          fill = type),
                      alpha = 0.6,
                      width = 0.3) +
        geom_text(data = ground_truth_formatted |>
                      filter(ubatch == "Batch 1" | ubatch == "Batch 2",
                             gpus == "1 GPU" | gpus == "2 GPUs" | gpus == "4 GPUs" | gpus == "8 GPUs"),
                  aes(x = -Inf,
                      y = Inf,
                      label = paste0("Ground Truth: ", round(reference, 2))),
                  hjust = -0.05,
                  vjust = 1.5,
                  size = 3,
                  color = "black") +
        geom_text(data = plot_data |>
                  filter(ubatch == "Batch 1" | ubatch == "Batch 2",
                         gpus == "1 GPU" | gpus == "2 GPUs" | gpus == "4 GPUs" | gpus == "8 GPUs"),
                  aes(x = type,
                      y = total_time_ms + total_time_ms_ci,
                      color = type,
                      label = round(total_time_ms, 2)),
                  vjust = 1.5, size = 3.4) +
        scale_color_brewer(palette = "Set1") +
        scale_fill_brewer(palette = "Set1") +
        guides(color = guide_legend(label = FALSE)) +
        facet_grid(ubatch ~ gpus, scales = "free_y") +
        labs(
            x = "Simulation Method",
            y = "Time (s)",
        ) +
        scale_y_continuous(expand = expansion(mult = c(0.25, 0.25))) +
        theme_bw(base_size = plot_base_size) +
        theme(
            legend.position = "none",
            strip.background = element_rect(fill = "white"),
            plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
        )

    p_ubatch_1_2_multi_node <- ggplot() +
        geom_rect(
            data = ground_truth_formatted |>
                filter(ubatch == "Batch 1" | ubatch == "Batch 2",
                       gpus != "1 GPU" & gpus != "2 GPUs" & gpus != "4 GPUs" & gpus != "8 GPUs"),
            aes(
                xmin = -Inf,
                xmax = Inf,
                ymin = reference - total_time_ms_ci,
                ymax = reference + total_time_ms_ci
            ),
            fill = "grey",
            alpha = 0.6,
            inherit.aes = FALSE
        ) +
        geom_hline(data = ground_truth_formatted |>
                       filter(ubatch == "Batch 1" | ubatch == "Batch 2",
                              gpus != "1 GPU" & gpus != "2 GPUs" & gpus != "4 GPUs" & gpus != "8 GPUs"),
                   aes(yintercept = reference),
                   linetype = "dashed",
                   color = "black") +
        geom_crossbar(data = plot_data |>
                          filter(ubatch == "Batch 1" | ubatch == "Batch 2",
                                 gpus != "1 GPU" & gpus != "2 GPUs" & gpus != "4 GPUs" & gpus != "8 GPUs"),
                      aes(x = type,
                          y = total_time_ms,
                          ymin = total_time_ms - total_time_ms_ci,
                          ymax = total_time_ms + total_time_ms_ci,
                          color = type,
                          fill = type),
                      alpha = 0.6,
                      width = 0.3) +
        geom_text(data = ground_truth_formatted |>
                      filter(ubatch == "Batch 1" | ubatch == "Batch 2",
                             gpus != "1 GPU" & gpus != "2 GPUs" & gpus != "4 GPUs" & gpus != "8 GPUs"),
                  aes(x = -Inf,
                      y = Inf,
                      label = paste0("Ground Truth: ", round(reference, 2))),
                  hjust = -0.05,
                  vjust = 1.5,
                  size = 3,
                  color = "black") +
        geom_text(data = plot_data |>
                      filter(ubatch == "Batch 1" | ubatch == "Batch 2",
                             gpus != "1 GPU" & gpus != "2 GPUs" & gpus != "4 GPUs" & gpus != "8 GPUs"),
                  aes(x = type,
                      y = total_time_ms + total_time_ms_ci,
                      color = type,
                      label = round(total_time_ms, 2)),
                  vjust = 1.5, size = 3.4) +
        scale_color_brewer(palette = "Set1") +
        scale_fill_brewer(palette = "Set1") +
        guides(color = guide_legend(label = FALSE)) +
        facet_grid(ubatch ~ gpus, scales = "free_y") +
        labs(
            x = "Simulation Method",
            y = "Time (s)",
        ) +
        scale_y_continuous(expand = expansion(mult = c(0.25, 0.25))) +
        theme_bw(base_size = plot_base_size) +
        theme(
            legend.position = "none",
            strip.background = element_rect(fill = "white"),
            plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
        )


    p_ubatch_4_8 <- ggplot() +
        geom_rect(
            data = ground_truth_formatted |> filter(ubatch == "Batch 8" | ubatch == "Batch 4"),
            aes(
                xmin = -Inf,
                xmax = Inf,
                ymin = reference - total_time_ms_ci,
                ymax = reference + total_time_ms_ci
            ),
            fill = "grey",
            alpha = 0.6,
            inherit.aes = FALSE
        ) +
        geom_hline(data = ground_truth_formatted |> filter(ubatch == "Batch 8" | ubatch == "Batch 4"),
                   aes(yintercept = reference),
                   linetype = "dashed",
                   color = "black") +
        geom_crossbar(data = plot_data |> filter(ubatch == "Batch 8" | ubatch == "Batch 4"),
                      aes(x = type,
                          y = total_time_ms,
                          ymin = total_time_ms - total_time_ms_ci,
                          ymax = total_time_ms + total_time_ms_ci,
                          color = type,
                          fill = type),
                      alpha = 0.6,
                      width = 0.3) +
        geom_text(data = ground_truth_formatted |> filter(ubatch == "Batch 8" | ubatch == "Batch 4"),
                  aes(x = -Inf,
                      y = Inf,
                      label = paste0("Ground Truth: ", round(reference, 2))),
                  hjust = -0.05,
                  vjust = 1.5,
                  size = 3,
                  color = "black") +
        geom_text(data = plot_data |> filter(ubatch == "Batch 8" | ubatch == "Batch 4"),
                  aes(x = type,
                      y = total_time_ms + total_time_ms_ci,
                      color = type,
                      label = round(total_time_ms, 2)),
                  vjust = 1.5, size = 3.4) +
        scale_color_brewer(palette = "Set1") +
        scale_fill_brewer(palette = "Set1") +
        guides(color = guide_legend(label = FALSE)) +
        facet_grid(ubatch ~ gpus, scales = "free_y") +
        labs(
            x = "Simulation Method",
            y = "Time (s)",
        ) +
        scale_y_continuous(expand = expansion(mult = c(0.25, 0.25))) +
        theme_bw() +
        theme(
            legend.position = "none",
            strip.background = element_rect(fill = "white"),
            plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
        )

    dir.create(dirname(output_file_batch_1_2_1_node), showWarnings = FALSE, recursive = TRUE)
    ggsave(output_file_batch_1_2_1_node, p_ubatch_1_2_1_node, width = width, height = height)

    dir.create(dirname(output_file_batch_1_2_multi_node), showWarnings = FALSE, recursive = TRUE)
    ggsave(output_file_batch_1_2_multi_node, p_ubatch_1_2_multi_node, width = width, height = height)

    dir.create(dirname(output_file_batch_4_8), showWarnings = FALSE, recursive = TRUE)
    ggsave(output_file_batch_4_8, p_ubatch_4_8, width = width / 2, height = height)

    return()
}

performance_comparison_publication_H100 <- function(input_file = "results/sc25/simulation_results/performance_ratios_H100_factorial_expanded.csv",
                                                    output_file_batch_1_2 = "results/sc25/figures/comparison_H100_batch_1_2.pdf",
                                                    width = 12,
                                                    height = 8) {
    df <- read_csv(input_file)

    ground_truth <- df %>%
        filter(type == "real") %>%
        select(gpus, ubatch, reference = total_time_ms, total_time_ms_ci)

    plot_data <- df %>%
        filter(type %in% c("roofline", "profiling", "model")) %>%
        left_join(ground_truth, by = c("gpus", "ubatch", "total_time_ms_ci")) %>%
        select(ubatch, gpus, type, total_time_ms, total_time_ms_ci, reference) %>%
        mutate(
            ubatch = factor(ubatch,
                            levels = c("1", "2", "4", "8"),
                            labels = c("Batch 1", "Batch 2", "Batch 4", "Batch 8")),
            gpus = factor(gpus,
                          levels = c("1", "2", "4", "8", "16",
                                     "32", "64", "128"),
                          labels = c("1 GPU", "2 GPUs", "4 GPUs", "8 GPUs",
                                     "16 GPUs", "32 GPUs", "64 GPUs", "128 GPUs")),
            type = factor(type,
                          levels = c("roofline", "profiling", "model"),
                          labels = c("roofline", "profiler", "model"))
        )

    ground_truth_formatted <- ground_truth %>%
        mutate(
            ubatch = factor(ubatch,
                            levels = c("1", "2", "4", "8"),
                            labels = c("Batch 1", "Batch 2", "Batch 4", "Batch 8")),
            gpus = factor(gpus,
                          levels = c("1", "2", "4", "8",
                                     "16", "32", "64", "128"),
                          labels = c("1 GPU", "2 GPUs", "4 GPUs", "8 GPUs",
                                     "16 GPUs", "32 GPUs", "64 GPUs", "128 GPUs")),
            )

    p_ubatch_1_2 <- ggplot() +
        geom_rect(
            data = ground_truth_formatted |> filter(ubatch == "Batch 1" | ubatch == "Batch 2"),
            aes(
                xmin = -Inf,
                xmax = Inf,
                ymin = reference - total_time_ms_ci,
                ymax = reference + total_time_ms_ci
            ),
            fill = "grey",
            alpha = 0.6,
            inherit.aes = FALSE
        ) +
        geom_hline(data = ground_truth_formatted |> filter(ubatch == "Batch 1" | ubatch == "Batch 2"),
                   aes(yintercept = reference),
                   linetype = "dashed",
                   color = "black") +
        geom_crossbar(data = plot_data |> filter(ubatch == "Batch 1" | ubatch == "Batch 2"),
                      aes(x = type,
                          y = total_time_ms,
                          ymin = total_time_ms - total_time_ms_ci,
                          ymax = total_time_ms + total_time_ms_ci,
                          color = type,
                          fill = type),
                      alpha = 0.6,
                      width = 0.3) +
        geom_text(data = ground_truth_formatted |> filter(ubatch == "Batch 1" | ubatch == "Batch 2"),
                  aes(x = -Inf,
                      y = Inf,
                      label = paste0("Ground Truth: ", round(reference, 2))),
                  hjust = -0.05,
                  vjust = 1.5,
                  size = 3,
                  color = "black") +
        geom_text(data = plot_data |> filter(ubatch == "Batch 1" | ubatch == "Batch 2"),
                  aes(x = type,
                      y = total_time_ms + total_time_ms_ci,
                      color = type,
                      label = round(total_time_ms, 2)),
                  vjust = 1.5, size = 3.4) +
        scale_color_brewer(palette = "Set1") +
        scale_fill_brewer(palette = "Set1") +
        guides(color = guide_legend(label = FALSE)) +
        facet_grid(ubatch ~ gpus, scales = "free_y") +
        labs(
            x = "Simulation Method",
            y = "Time (s)",
        ) +
        scale_y_continuous(expand = expansion(mult = c(0.25, 0.25))) +
        theme_bw(base_size = 14) +
        theme(
            legend.position = "none",
            strip.background = element_rect(fill = "white"),
            plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
        )

    dir.create(dirname(output_file_batch_1_2), showWarnings = FALSE, recursive = TRUE)
    ggsave(output_file_batch_1_2, p_ubatch_1_2, width = width, height = height)

    return()
}

# Find and load simulation files
all_files <- find_all_files(
    path = "results/sc25/astra_sim_measurements/A100_factorial_expanded",
    pattern = "EndToEnd.csv"
)

# Load files in batches
df_sim_full <- load_files_in_batches(all_files, batch_size = 500)

# Format simulation data
df_sim <- format_simulation_data(df_sim_full)

# Load and format real data
df_real <- load_and_format_real_data(list(
    path = "results/lit-llama/block-activation-checkpointing/",
    pattern = "A100"
))

# Combine, adjust and save the data
df <- adjust_data_and_save(
    df_sim,
    df_real,
    "results/sc25/astra_sim_measurements/aggregated_A100_factorial_expanded.csv"
)

df <- load_performance_data("results/sc25/astra_sim_measurements/aggregated_A100_factorial_expanded.csv")

ref_df <- create_reference_dataframe(df)

ratio_df <- calculate_performance_ratios(df, ref_df)
save_and_print_ratios(ratio_df, "results/sc25/simulation_results/performance_ratios_A100_factorial_expanded.csv")
print(ratio_df, n = Inf)

# Fitting coefficient curve for models
ratio_df <- load_performance_data("results/sc25/simulation_results/performance_ratios_A100_factorial_expanded.csv")
df_lm_model <- prepare_model_data(ratio_df, "model", 1)
model_fit <- fit_coefficient_model(df_lm_model)

# Fitting coefficient curve for profiling
df_lm_profiling <- prepare_model_data(ratio_df, "profiling", 1)
profiling_fit <- fit_coefficient_model(df_lm_profiling)

coefficient_trend_plots_publication(
    input_file = "results/sc25/simulation_results/performance_ratios_A100_factorial_expanded.csv",
    output_file = "results/sc25/figures/coefficient_faceted_trends_A100_factorial_expanded_publication.pdf",
    grid_size = 32,
    max_ubatch = 2,
    max_gpus = 128,
    output_width = 8,
    output_height = 3.5
)

performance_comparison_publication_A100(width = 10,
                                        height = 3)

performance_comparison_publication_H100(width = 10,
                                        height = 4)

library(readr)

df = read_csv("results/sc25/model_training_summary/a100_mean_factorial_expanded_specific/all_kernels_model_evaluation.csv",
              show_col_types = FALSE) |>
    filter(gpu == "a100_mean_factorial_expanded") |>
    mutate(gpu = "A100")

library(tidyverse)
library(knitr)

# Function to format scientific notation with proper LaTeX escaping
format_scientific_latex <- function(x, digits = 2) {
  if(is.na(x) || is.null(x)) return("NA")

  # Parse the number into mantissa and exponent
  exponent <- floor(log10(abs(x)))
  mantissa <- x / 10^exponent

  # Round mantissa to specified digits
  mantissa <- round(mantissa, digits)

  # Format as LaTeX with DOUBLE escaping for backslashes
  return(sprintf("$%.2f \\times 10^{%d}$", mantissa, exponent))
}

# Create the formatted dataframe
df_formatted <- df %>%
  # Create combined y range column with properly escaped LaTeX math symbols
  mutate(test_y_range = sprintf("$y \\in [%.5f, %.5f], \\mu = %.5f$",
                               min_test_y, max_test_y, mean_test_y),
         out_of_confidence_pct = out_of_confidence * 100,
         lin_reg_RMSE_formatted = sapply(lin_reg_RMSE, format_scientific_latex),
         MAE_formatted = sapply(MAE, format_scientific_latex),
         RMSE_formatted = sapply(RMSE, format_scientific_latex),
         name = gsub("_", "\\\\_", name)) %>%
  # Select and arrange columns - grouping similar metrics together
  select(name, gpu,
         R2, lin_reg_R2,
         RMSE_formatted, lin_reg_RMSE_formatted,
         out_of_confidence_pct,
         test_y_range)

# Create a simple LaTeX table manually without using kable
latex_rows <- apply(df_formatted, 1, function(row) {
  sprintf("%s & %s & %.4f & %.4f & %s & %s & %.4f & %s \\\\",
          row["name"], row["gpu"],
          as.numeric(row["R2"]), as.numeric(row["lin_reg_R2"]),
          row["RMSE_formatted"], row["lin_reg_RMSE_formatted"],
          as.numeric(row["out_of_confidence_pct"]),
          row["test_y_range"])
})

# Create complete LaTeX document (figure-style)
latex_document <- c(
  "\\documentclass[border=10pt]{standalone}",
  "\\usepackage{booktabs}",
  "\\usepackage{array}",
  "\\usepackage{amsmath}",
  "\\usepackage[table]{xcolor}",
  "\\begin{document}",
  "",
  "\\begin{tabular}{llrrrrrr}",
  "\\toprule",
  "\\textbf{Kernel} & \\textbf{GPU} & \\textbf{R$^2$} & \\textbf{Lin. Reg. R$^2$} & \\textbf{RMSE} & \\textbf{Lin. Reg. RMSE} & \\textbf{OOC (\\%)} & \\textbf{Test} $y$ \\textbf{Range} \\\\",
  "\\midrule",
  latex_rows,
  "\\bottomrule",
  "\\end{tabular}",
  "",
  "\\end{document}"
)

# Write to file
writeLines(paste(latex_document, collapse = "\n"), "results/sc25/figures/tmp_table_A100_factorial_expanded.tex")

# Now compile to PDF and move to the figures directory
system("pdflatex -output-directory=results/sc25/figures results/sc25/figures/tmp_table_A100_factorial_expanded.tex")
system("mv results/sc25/figures/tmp_table_A100_factorial_expanded.pdf results/sc25/figures/model_performance_table_A100_factorial_expanded.pdf")
system("rm results/sc25/figures/tmp_table_A100_factorial_expanded.aux")

cat("PDF generated and saved to results/sc25/figures/model_performance_table_A100_factorial_expanded.pdf\n")

library(tidyverse)
library(scales)
library(RColorBrewer)

h100_specs <- tibble(
    gpu = "NVIDIA H100",
    peak_fp32_tflops = 989,
    peak_fp32_standard_tflops = 67,
    memory_bandwidth_gbs = 3350,
    ridge_point_tensor = peak_fp32_tflops / memory_bandwidth_gbs,
    ridge_point_standard = peak_fp32_standard_tflops / memory_bandwidth_gbs
)

ai_values <- tibble(
    arithmetic_intensity = 10^seq(-4, 2.5, by = 0.05)
)

roofline_data <- ai_values %>%
    mutate(
        memory_bound_perf = arithmetic_intensity * h100_specs$memory_bandwidth_gbs,
        compute_bound_tensor = h100_specs$peak_fp32_tflops,
        compute_bound_standard = h100_specs$peak_fp32_standard_tflops,
        attainable_perf_tensor = pmin(memory_bound_perf, compute_bound_tensor),
        attainable_perf_standard = pmin(memory_bound_perf, compute_bound_standard)
    )

set1_colors <- brewer.pal(9, "Set1")[1:3]

roofline_plot <- ggplot() +
    geom_line(data = roofline_data |>
                  filter(arithmetic_intensity <= h100_specs$ridge_point_tensor),
              aes(x = arithmetic_intensity,
                  y = memory_bound_perf,
                  color = "Memory Bandwidth Limit"),
              size = 1) +
    geom_hline(aes(yintercept = h100_specs$peak_fp32_standard_tflops,
                   color = "Standard FP32 Peak Performance"),
               size = 1) +
    geom_hline(aes(yintercept = h100_specs$peak_fp32_tflops,
                   color = "Tensor Cores Peak Performance"),
               size = 1) +
    geom_point(data = h100_specs,
               aes(x = ridge_point_standard,
                   y = peak_fp32_standard_tflops,
                   color = "Memory Bandwidth Limit"),
               show.legend = FALSE,
               size = 3) +
    geom_point(data = h100_specs,
               aes(x = ridge_point_tensor,
                   y = peak_fp32_tflops,
                   color = "Memory Bandwidth Limit"),
               show.legend = FALSE,
               size = 3) +
    scale_x_log10(
        breaks = 10^seq(-4, 5, 1),
        labels = trans_format("log10", math_format(10^.x)),
        limits = c(10^-2.5, 10^2.5)
    ) +
    scale_y_log10(
        limits = c(20, 1200),
        breaks = c(20, 60, 100, 500, 989),
        labels = c("20", "60", "100", "500", "989")
    ) +
    scale_color_manual(
        values = c(
            "Memory Bandwidth Limit" = set1_colors[1],
            "Standard FP32 Peak Performance" = set1_colors[2],
            "Tensor Cores Peak Performance" = set1_colors[3])
    ) +
    labs(
        x = "Arithmetic Intensity (FLOP/Byte)",
        y = "Attainable Performance (TFLOP/s)",
        color = "Performance Limits"
    ) +
    annotate("text",
             x = h100_specs$ridge_point_standard * 2.5,
             y = h100_specs$peak_fp32_standard_tflops * 0.98,
             label = "Ridge Point\n(Standard FP32)",
             size = 3.5) +
    annotate("text",
             x = h100_specs$ridge_point_tensor * 2.5,
             y = h100_specs$peak_fp32_tflops * 0.99,
             label = "Ridge Point\n(Tensor Cores)",
             size = 3.5) +
    theme_bw() +
    theme(
        legend.position = c(0.8, 0.5),
        axis.title = element_text(size = 12),
        legend.title = element_text(size = 11)
    )

example_apps <- tibble(
    name = c("Tensor Compute Bound App.", "Memory Bound App.", "Compute Bound App.", "Tensor Memory Bound App."),
    arithmetic_intensity = c(100, 0.014, 1.5, 0.1),
    performance = c(800, 40, 55, 290)
)

roofline_plot_with_apps <- roofline_plot +
    geom_point(data = example_apps,
               aes(x = arithmetic_intensity, y = performance),
               color = brewer.pal(9, "Set1")[4], size = 4, shape = 17) +
    # geom_text(data = example_apps,
    #           aes(x = arithmetic_intensity, y = performance * 0.82,
    #               label = name),
    #           size = 3.5)
    geom_label(data = example_apps,
              aes(x = arithmetic_intensity, y = performance * 0.82,
                  label = name),
              size = 3.5,
              fill = rgb(0, 0, 0, 0.4),
              color = "white",
              label.padding = unit(0.15, "lines"),
              label.size = 0)  # Removes border

ggsave("results/sc25/figures/h100_roofline_model.pdf", roofline_plot,
       width = 10, height = 7, dpi = 300)
ggsave("results/sc25/figures/h100_roofline_model_with_apps.pdf", roofline_plot_with_apps,
       width = 7.5, height = 4, dpi = 300)
