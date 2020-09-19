# Simplified Epinions Dataset

The Epinions dataset is a classic dataset often used in the evaluation of social recommendation algorithms. Since the original Epinions dataset is a little bit large, we used a simplified version of Epinions dataset in the experiment of Chapter 10, which contains 3139 users and 8145 items. The dataset is cropped from the original Epinions dataset.

# Original Epinions Dataset
The following is a description of the original Epinions dataset. (You can check it out at http://www.trustlet.org/epinions.html)

We have collected and released 2 different versions of Epinions datasets:

Downloaded Epinions dataset
Extended Epinions dataset
Epinions is a website where people can review products. Users can register for free and start writing subjective reviews about many different types of items (software, music, television show, hardware, office appliances, ...). A peculiar characteristics of Epinions is that users are paid according to how much a review is found useful (Income Share program).

Also because of that, the attempts to game the systems are many and, as a possible fix, a trust system was put in place. Users can add other users to their "Web of Trust", i.e. "reviewers whose reviews and ratings they have consistently found to be valuable" and their "Block list", i.e. "authors whose reviews they find consistently offensive, inaccurate, or in general not valuable" (see the explanation of Epinions Web of Trust as backupped by Archive.org).

The dataset was collected by Paolo Massa in a 5-week crawl (November/December 2003) from the Epinions.com Web site.

The dataset contains

* 49,290 users who rated a total of
* 139,738 different items at least once, writing
* 664,824 reviews and
* 487,181 issued trust statements.
Users and Items are represented by anonimized numeric identifiers.

The dataset consists of 2 files.

ratings_data.txt.bz2 (2.5 Megabytes): it contains the ratings given by users to items.

Every line has the following format:

<pre>user_id item_id rating_value
</pre>
For example,
<pre>23 387 5
</pre>
represents the fact "user 23 has rated item 387 as 5"

Ranges:

* user_id is in [1,49290]
* item_id is in [1,139738]
* rating_value is in [1,5]

trust_data.txt.bz2 (1.7 Megabytes): it contains the trust statements issued by users.

Every line has the following format:

<pre>source_user_id target_user_id trust_statement_value
</pre>
For example, the line
<pre>22605 18420 1
</pre>
represents the fact "user 22605 has expressed a positive trust statement on user 18420"

Ranges:

* source_user_id and target_user_id are in [1,49290]
* trust_statement_value is always 1 (since in the dataset there are only positive trust statements and not negative ones (distrust)).

Note: there are no distrust statements in the dataset (block list) but only trust statements (web of trust), because the block list is kept private and not shown on the site.

