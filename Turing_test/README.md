1. Compress images with preprocessing/compress2jpg.py.
2. Get the last frames with analyze/last_frames.py.
3. Get the left and right crops with analyze/crop_left_and_right.py.
4. Encode images with analyze/img2feat/encode2real_feat_recon.py.
5. Select images for Turing test with Turing_test/image_selection.py.
   For stylegan2-ada comparison:
6. Use project_all.sh to get the reconstructions.
7. Use Turing_test/mv_bp.py to collect the reconstructed images.
8. Use Turing_test/gen_images.py to generate random images.

After obtaining the images, follow these steps to build a Turing test in Qualtrics:

1. Upload images: Upload all images to the Library, preferably organized into separate folders.
2. Create a project: Start a new project and use Loop & Merge to generate multiple questions with the same format. Use
   ${lm://Field/1} to indicate the fields that will change in each quiz.
3. Configure Loop & Merge: Create a table in Loop & Merge with columns equal to the number of fields and rows equal to
   the number of quizzes.
4. Add images: To insert images, use the HTML format: \<img src="img_url" style="width:256px;height:256px;"/>. Replace "
   img_url" with the actual image URL. To find the URL of the uploaded images, follow these steps:

- Go to Account Settings -> Qualtrics ID.
- Click on the Library where the images are uploaded, under the Library section.
- Scroll down to the folder containing the images.
- Select the graphics name and ID and paste them into an Excel sheet, creating two columns.
- Use Excel's concatenate formula to generate URLs in this
  format: https://YOUR_DATACENTER_ID.qualtrics.com/ControlPanel/Graphic.php?IM=IMAGE_ID

5. Add ${lm://CurrentLoopNumber} in the question stem. Go to left Survey flow -> Add a New Element Here (on the top) ->
   Set Embedded Data -> index = ${lm://CurrentLoopNumber}
6. Generate test responses using Tools->Generate test responses.