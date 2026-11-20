// 放在仓库根目录。Jenkins 每 5 分钟检查一次 main，有新 commit 就跑测试
pipeline {
    agent any

    triggers {
        pollSCM('H/5 * * * *')   // 本机 Jenkins 收不到 GitHub webhook，用轮询代替
    }

    options {
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timestamps()
    }

    environment {
        MUJOCO_GL = 'disable'                          // 测试不渲染，关闭 OpenGL
        PIP_CACHE_DIR = "${env.JENKINS_HOME}/.cache/pip" // 跨构建复用 pip 缓存，第二次起安装更快
    }

    stages {
        stage('Setup') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install torch --index-url https://download.pytorch.org/whl/cpu
                    pip install -r requirements-ci.txt
                '''
            }
        }

        stage('Test') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python -m pytest -q --junitxml=reports/junit.xml
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/junit.xml'  // 在 Jenkins 页面显示测试趋势图
        }
    }
}
