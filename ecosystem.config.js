// module.exports = {
//     apps : [{
//       name: "index-faces",
//       script: "/home/ec2-user/face-recognition-services/ve-ai-face-recognition-face-index/src/main.py",
//       interpreter: "/home/ec2-user/face-recognition-services/ve-ai-face-recognition-face-index/venvIndexFaces/bin/python3",
//       instances: 1,
//       exec_mode: "fork",
//       watch: false,
//       max_memory_restart: "2G",
//       env: {
//         "NODE_ENV": "production"
//       }
//     },
//     {
//       name: "index-faces-2",
//       script: "/home/ec2-user/face-recognition-services/ve-ai-face-recognition-face-index/src/main.py",
//       interpreter: "/home/ec2-user/face-recognition-services/ve-ai-face-recognition-face-index/venvIndexFaces/bin/python3",
//       instances: 1,
//       exec_mode: "fork",
//       watch: false,
//       max_memory_restart: "2G",
//       env: {
//         "NODE_ENV": "production"
//       }
//     },
//     {
//       name: "index-faces-3",
//       script: "/home/ec2-user/face-recognition-services/ve-ai-face-recognition-face-index/src/main.py",
//       interpreter: "/home/ec2-user/face-recognition-services/ve-ai-face-recognition-face-index/venvIndexFaces/bin/python3",
//       instances: 1,
//       exec_mode: "fork",
//       watch: false,
//       max_memory_restart: "2G",
//       env: {
//         "NODE_ENV": "production"
//       }
//     }
//     ]
//   }


//Create a new app for the face index service to run the service on port 8000
module.exports = {
    apps : [{
      name: "index-faces",
      script: "/home/ec2-user/face-recognition-services/ve-ai-face-recognition-face-index/src/main.py",
      interpreter: "/home/ec2-user/face-recognition-services/ve-ai-face-recognition-face-index/venvIndexFaces/bin/python3",
      instances: 1,
      exec_mode: "fork",
      watch: false,
      max_memory_restart: "5G",
      env: {
        "NODE_ENV": "production"
      }
    }]
  }