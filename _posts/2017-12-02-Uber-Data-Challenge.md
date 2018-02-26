---
layout: post
image: '/img/uber.png'
title: "Solving the Uber Data Challenge"
subtitle: "Category: Interview Prep (10 min read)"
---

No matter what field you're in or what role you're aiming for it seems like the interview process is difficult for everyone.  As a data scientist, during the interview process you’re often expected to analyze data, implement machine learning models, and/or use statistical techniques to derive insights from sample datasets.  In this post I’ll go over my approach to tackling Uber’s data scientist take-home exercise in the hopes of helping future data scientist applicants better prepare for their interview process.  I’ll also try to give helpful insights along the way (I went through the process last summer and flew out for the final interview, but ultimately didn’t get the job).  You can check out my Github repo for their exercise problems along with my solutions.  There were two parts to the exercise and applicants are given one week to submit their solutions.  In addition to the exercise, data scientist applicants can expect to go through at least two phone screenings, one of which should be technically oriented.  

The first part of their data exercise was about SQL syntax and contained two subproblems.  They give you made-up tables, but no data.  Since I was a little worried about the possibility of including a syntactic error in one of my SQL statements, I created a PostgreSQL database (with the requisite tables) and then filled it in with simulated data to test out my queries.

![png](/img/project-7_files/query.png)
*Figure 1 - Screen capture of my PostgreSQL database query using simulated data.*

The second part of their data exercise was more open-ended (with three subproblems) and involved data cleaning, exploratory analysis, visualizations and machine learning modeling.  They gave me a sample dataset that contained information about driver signups (e.g., vehicle type, city, driver signup date, etc.).  I was tasked with building machine learning models to predict whether or not a driver signup would, in fact, become an actual driver (i.e., complete their first trip).  They also wanted me to explore keys features that could impact driver signup conversions, as well as any additional business insights could be derived from the data.  I tested 8 different binary classification models out and ultimately the gradient boosting model as the best performing model.  

After submitting the data exercise, your submission is graded by someone from outside the team you’re interviewing for (or at least that’s my understanding).  Following the data exercise there may be an additional round of a phone interview.  During the final round of interviews you will probably be asked to present your findings from the data exercise, at which point they will probably ask specific questions about each of the models (including model derivations) as well as the limitations/caveats associate with each of them.  

I hope this post is helpful for data scientist applicants who are currently going through Uber’s interview process or potential future applicants.  

![png](/img/project-7_files/image1.png)
*Figure 2 - Blah blah blah.*

![png](/img/project-7_files/image2.png)
*Figure 3 - Blah blah blah.*

![png](/img/project-7_files/image3.png)
*Figure 4 - Blah blah blah.*

![png](/img/project-7_files/image4.png)
*Figure 5 - Blah blah blah.*

![png](/img/project-7_files/image5.png)
*Figure 6 - Blah blah blah.*

![png](/img/project-7_files/image6.png)
*Figure 7 - Blah blah blah.*

Above are the data exercise slides I presented during my final interview -- I hope they can serve as a good starting point for your own analysis.  Good luck!


