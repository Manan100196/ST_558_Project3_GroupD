library(rmarkdown)

out_params <- c("lifestyle","entertainment","bus","socmed","tech","world")
out_params_list = lapply(out_params, FUN = function(x){list(var = x)})

out_filename <- c("Lifestyle_analysis.md","Entertainment_analysis.md",
                  "Business_analysis.md","Social_media_analysis.md",
                  "Tech_analysis.md","World_analysis.md")
for (i in 1:6){
  rmarkdown::render(input = "main.Rmd", output_file = out_filename[i],
                    params = out_params_list[[i]])
}