/*
navigator.getUserMedia
navigator.getUserMedia is now deprecated and is replaced by navigator.mediaDevices.getUserMedia. To fix this bug replace all versions of navigator.getUserMedia with navigator.mediaDevices.getUserMedia

Low-end Devices Bug
The video eventListener for play fires up too early on low-end machines, before the video is fully loaded, which causes errors to pop up from the Face API and terminates the script (tested on Debian [Firefox] and Windows [Chrome, Firefox]). Replaced by playing event, which fires up when the media has enough data to start playing.
*/

const video = document.getElementById('video')


console.log("Loading...");
Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('/Face-Api/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/Face-Api/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/Face-Api/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/Face-Api/models'),
    // faceapi.nets.faceExpressionNet.loadFromUri('/Face-Api/models')
]).then(start)


function startVideo() {
    console.log("Start");
    navigator.getUserMedia(
        { video: {} },
        stream => video.srcObject = stream,
        err => console.error(err)
    )
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


async function start() {
    console.log("Training data...")
    const labeledFaceDescriptors = await loadLabeledImages();
    console.log("Completed training.")

    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

    alert('Completed training.');

    startVideo();

    video.addEventListener('playing', () => {
        const canvas = faceapi.createCanvasFromMedia(video)
        document.body.append(canvas)
    
        const displaySize = { width: video.width, height: video.height }
        faceapi.matchDimensions(canvas, displaySize)
    
        setInterval(async () => {
            const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors();
            const resizedDetections = faceapi.resizeResults(detections, displaySize)
    
            canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    
            const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));
            
            results.forEach((result, i) => {
                const box = resizedDetections[i].detection.box;
                const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
                drawBox.draw(canvas);
            })
            // faceapi.draw.drawDetections(canvas, resizedDetections)
            // faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
            // faceapi.draw.drawFaceExpressions(canvas, resizedDetections)
        }, 100)
    })
}


