# ST_558_Project3_GroupD

This repo was created for Project 3 of our curriculum for the course ST 558 (Fall 2022). This is project involves creating predictive models and automating Markdown reports.
Here we are working with [online news popularity dataset](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).  
We have subset the data by **data_channel_is_\*** to create six subsets based on the type of article. The types of articles include: Lifestyle, Entertainment, Business, Social Media, Tech and World.  
We have summarized the data and try to predict the number of **shares** using predictive models for each of the article type.  

---
### List of R packages used
1. tidyverse
2. caret
3. kableExtra
4. corrplot
5. timereg
6. rmarkdown

---
### Link to generated analyses
You can find the links to the generated analyses for each article type here:  
[Lifestyle articles](Lifestyle_analysis.html)  
[Entertainment articles](Entertainment_analysis.html)  
[Business articles](Business_analysis.html)  
[Social Media articles](Social_media_analysis.html)  
[Tech articles](Tech_analysis.html)  
[World articles](World_analysis.html)  

---
### Code to generate analysis file automatically
`out_params` is a vector of parameters which are used as input parameters to **main.Rmd** file. Since the `param` argument in `render()` function requires params to be a named list, we have converted each element of `out_params` to a named list using `lapply()` into `out_params_list`.  

`out_filename` is a vector of output file names we want for each of the input parameter.  

Since there are only six parameters, we decided to go with a for loop to iterate over the file names and vector of named lists to create six different .md files based on the type of article, namely:
 - Lifestyle_analysis.md
 - Entertainment_analysis.md
 - Business_analysis.md
 - Social_media_analysis.md
 - Tech_analysis.md
 - World_analysis.md

```
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
```
