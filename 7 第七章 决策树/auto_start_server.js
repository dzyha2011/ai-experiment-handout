#!/usr/bin/env node
/**
 * 自动启动Flask服务器脚本
 * 用于在网页中自动启动watermelon_server.py
 */

const { spawn, exec } = require('child_process');
const path = require('path');
const fs = require('fs');

// 获取当前脚本所在目录
const scriptDir = __dirname;
const serverFile = path.join(scriptDir, 'watermelon_server.py');

console.log('='.repeat(50));
console.log('决策树实验 - Flask服务器自动启动脚本');
console.log('='.repeat(50));
console.log(`当前目录: ${scriptDir}`);
console.log(`服务器文件: ${serverFile}`);

// 检查服务器文件是否存在
if (!fs.existsSync(serverFile)) {
    console.error(`错误: 找不到服务器文件 ${serverFile}`);
    process.exit(1);
}

// 检查Python是否可用
function checkPython() {
    return new Promise((resolve) => {
        exec('python --version', (error, stdout, stderr) => {
            if (error) {
                exec('python3 --version', (error3, stdout3, stderr3) => {
                    resolve(error3 ? null : 'python3');
                });
            } else {
                resolve('python');
            }
        });
    });
}

// 检查端口是否被占用
function checkPort(port) {
    return new Promise((resolve) => {
        const net = require('net');
        const server = net.createServer();
        
        server.listen(port, () => {
            server.once('close', () => {
                resolve(false); // 端口未被占用
            });
            server.close();
        });
        
        server.on('error', () => {
            resolve(true); // 端口被占用
        });
    });
}

// 启动Flask服务器
async function startFlaskServer() {
    try {
        // 检查端口5000是否被占用
        const portInUse = await checkPort(5000);
        if (portInUse) {
            console.log('✅ Flask服务器可能已经在运行 (端口5000被占用)');
            return;
        }
        
        // 检查Python命令
        const pythonCmd = await checkPython();
        if (!pythonCmd) {
            console.error('❌ 错误: 找不到Python解释器');
            console.error('请确保已安装Python并添加到PATH环境变量');
            return;
        }
        
        console.log(`✅ 找到Python解释器: ${pythonCmd}`);
        console.log('🚀 正在启动Flask服务器...');
        console.log('📍 服务器地址: http://127.0.0.1:5000');
        console.log('⚠️  请保持此窗口打开，关闭窗口将停止服务器');
        console.log('-'.repeat(50));
        
        // 启动Python服务器
        const serverProcess = spawn(pythonCmd, ['watermelon_server.py'], {
            cwd: scriptDir,
            stdio: 'inherit'
        });
        
        // 处理服务器进程事件
        serverProcess.on('error', (error) => {
            console.error('❌ 启动失败:', error.message);
        });
        
        serverProcess.on('exit', (code, signal) => {
            if (code !== null) {
                console.log(`\n📊 服务器进程退出，退出码: ${code}`);
            } else if (signal) {
                console.log(`\n📊 服务器进程被信号终止: ${signal}`);
            }
        });
        
        // 处理进程终止信号
        process.on('SIGINT', () => {
            console.log('\n🛑 收到终止信号，正在关闭服务器...');
            serverProcess.kill('SIGINT');
            process.exit(0);
        });
        
        process.on('SIGTERM', () => {
            console.log('\n🛑 收到终止信号，正在关闭服务器...');
            serverProcess.kill('SIGTERM');
            process.exit(0);
        });
        
    } catch (error) {
        console.error('❌ 启动过程中发生错误:', error.message);
    }
}

// 如果直接运行此脚本
if (require.main === module) {
    startFlaskServer();
}

// 导出函数供其他模块使用
module.exports = {
    startFlaskServer,
    checkPython,
    checkPort
};