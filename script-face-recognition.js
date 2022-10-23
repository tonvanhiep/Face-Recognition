const imageUpload = document.getElementById('imageUpload');

console.log("Loading model...");

Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/Face-Api/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/Face-Api/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/Face-Api/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/Face-Api/models')
]).then(start);

async function start() {
    const container = document.createElement('div');
    container.style.position = 'relative';
    document.body.append(container);

    console.log("Training data...")
    const labeledFaceDescriptors = await loadLabeledImages();
    console.log("Completed training.")

    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

    let image;
    let canvas;

    document.body.append('Completed training.');

    imageUpload.addEventListener('change', async () => {
        console.log("Uploaded image")
        if (image) image.remove();
        if (canvas) canvas.remove();

        image = await faceapi.bufferToImage(imageUpload.files[0]);

        container.append(image);

        canvas = faceapi.createCanvasFromMedia(image);

        container.append(canvas);

        const displaySize = { width: image.width, height: image.height };
        faceapi.matchDimensions(canvas, displaySize);
        
        console.log("Processing...")
        const detections = await faceapi.detectAllFaces(image, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptors();
        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));
        
        results.forEach((result, i) => {
            const box = resizedDetections[i].detection.box;
            const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
            drawBox.draw(canvas);
        })
        console.log("Finish")
    })
}

function loadLabeledImages() {
    const labels = ['Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Thor', 'Ton_Van_Hiep', 'Tony Stark']
    return Promise.all(
        labels.map(async label => {
            const descriptions = [];
            for (let i = 1; i <= 2; i++) {
                const img = await faceapi.fetchImage(`./labeled_images/${label}/${i}.jpg`);
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                descriptions.push(detections.descriptor);
            }

            return new faceapi.LabeledFaceDescriptors(label, descriptions);
        })
    )
}
