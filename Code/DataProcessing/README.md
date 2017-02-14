# Reddit Data Processing

Data folder : `~/scratch/DERP/Reddit/`

### Script Files

 - `processData.py` : Contains some basic functions for cleaning raw file and generating the child-list dictionary

### Data Files

 - `RC_2015-01` : Raw reddit comments for 1 month. 54M comments.
 - `Comments.txt` : All 54M comments from the raw file with only the required headers of ['name', 'body', 'author', 'subreddit', 'parent_id', 'link_id', 'children']. Each line of file is a json string.
 - `Comments_NoChildren.txt` : Same as _Comments.txt_ but without 'children' header.
 - `UsefulComments.txt` : 38M useful comments found by removing top-level comments that do not have any children.
 - `UsefulComments_NoChildren.txt` : Analogous
 - `CommentsToObserve.pkl` : < To be updated >
 - `CommentsToMatch.pkl` : < To be updated >
 - `ChildrenDict.pkl` : Dictionary mapping comment 'name' to list of children 'name's
 - `ChildHistCount.pkl` : Dictionary counting the number of comments having 'n' child comments
 - `MultiReplyExamples` : Files containing examples of alternate child responses to same comment. #ar_cmt

